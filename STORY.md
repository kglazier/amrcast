# Why I built AMRCast

> The short version: I'm a senior software engineer with a biology degree. AMRCast is
> where those two halves meet — a production-quality tool that predicts how resistant a
> bacterial isolate is to antibiotics, built with the engineering discipline I'd apply to
> any production system and the domain judgment to know what *not* to build.

If you're evaluating me as an engineer, this document is the part of the repo worth
reading. The [README](README.md) tells you *what* AMRCast does and how to run it. This one
tells you *how I made decisions* — the scoping calls, the things I deleted, and the
mistakes I learned from. That's the signal I'd want from a portfolio piece, so it's the
signal I tried to put in.

## The problem, in one paragraph

When a patient has a serious bacterial infection, a clinician needs to know which
antibiotics will still work. The lab answer is an **MIC** — the minimum inhibitory
concentration, the lowest drug dose that stops the bug from growing. Measuring it in a lab
takes days of culturing. Meanwhile, we can sequence a bacterial genome in hours. So the
question is: *can you predict the MIC straight from the genome?* For tuberculosis, a large
consortium has largely solved this. For the everyday pathogens that fill hospitals —
*E. coli*, *Salmonella*, *Klebsiella* — there was no maintained, usable tool. That gap is
the project.

## Why this project, for this career move

I wanted one artifact that proved three things at once, because the roles I'm aiming for
need all three:

1. **I can build and ship real software** — not a notebook that runs once on my machine,
   but a packaged CLI with a layered architecture, a test suite, and a reproducible data
   pipeline.
2. **I can reason about a scientific domain** — frame the right problem, choose the right
   validation metric, and read a results table critically instead of celebrating it.
3. **I know where ML actually belongs** — most of this problem is *not* a modeling problem,
   and the engineering judgment is in knowing which 10% is.

A toy classifier on a Kaggle dataset proves none of those. A maintained tool that fills a
genuine gap in a field I understand proves all three.

## The decisions I'm most proud of (all of them are subtractions)

Good engineering on this project looked less like writing clever code and more like
refusing to write code I didn't need. Three calls in particular:

### 1. Don't reinvent gene detection — stand on AMRFinderPlus

Detecting resistance genes and point mutations in a genome is a *solved* problem. NCBI's
AMRFinderPlus is curated, organism-aware, and maintained by people whose full-time job is
keeping it current. I could have spent months building a worse version of it. Instead,
AMRFinderPlus is an input: it does gene detection, and AMRCast builds the **quantitative
prediction and interpretation layer** on top. The architecture reflects this — `genome/`
shells out to AMRFinderPlus, `features/` turns its output into model inputs, and the ML
never touches raw sequence. Knowing where the boundary of your system should be is the
whole job.

### 2. Predict a number, not a label — and explain it

Most existing tools output binary Resistant/Susceptible. That's the easier problem and the
crowded one. AMRCast predicts the actual MIC on a log2 scale and *then* derives the
S/I/R category from clinical (CLSI) breakpoints — which means the output carries more
information and degrades gracefully near the cutoff. On top of that, every prediction comes
with a SHAP explanation naming the specific genes and mutations that drove it (e.g.
`gyrA_S83L` pushing ciprofloxacin up by +3.84 log2). A black-box number is useless to a
microbiologist; an explained number is a tool they might actually trust. The interpretation
layer (`explain/`) was a deliberate scope choice, not an afterthought.

### 3. Know what's out of scope and stay out

The README has a "What this is NOT" section, and I treat it as a feature. Not a gene
detector. Not a binary classifier. Not for TB — the CRyPTIC consortium already solved that
with more data and more domain experts than I'll ever have. Scoping a project *down* to the
part that's genuinely unsolved is, in my experience, the most underrated senior skill, and
it's the one I most wanted this repo to demonstrate.

## The mistake that taught me the most

Early on, I did the obvious thing: I pulled in every public dataset I could find to maximize
training volume. More data is better, right? I trained on a large mixed-source dataset
(BV-BRC) and got **47% essential agreement** — barely better than guessing. I almost
concluded the whole approach didn't work.

The problem wasn't the model. It was that different labs measure MICs on different platforms
with different protocols, and mixing them injects more noise than signal. When I retrained
the *same model* on NARMS data — where every isolate is tested with standardized Sensititre
broth microdilution — essential agreement jumped to **93.7%**.

Same architecture, same code. The entire difference was data discipline. That experience
rewired how I approach ML problems: **data quality beats model complexity, almost every
time**, and the highest-leverage work is usually upstream of the model, not in it. It's the
single most transferable lesson in the project, and it's why I'd rather curate a clean
dataset than tune hyperparameters.

## What the results say — and what they don't

The headline numbers are strong: across *E. coli* (28 antibiotics, ~10.6k isolates),
*Salmonella* (12 antibiotics), and *Klebsiella* (26 antibiotics), most drugs clear 90%
essential agreement under 5-fold cross-validation, several above 97%. The full tables are
in the README.

I want to be precise about the limits, because reading a results table honestly is part of
the skill I'm claiming:

- **Some drugs are genuinely hard.** Extended-spectrum cephalosporins (cefepime, cefotaxime)
  lag, partly because resistance hinges on specific enzyme variants and partly because the
  training sets for those drugs are small. *Klebsiella* is a harder target across the board —
  complex carbapenem-resistance mechanisms and less data.
- **Cross-validation can flatter you.** Bacterial isolates are related by descent, so a
  naive random split can leak near-duplicate genomes across the train/test boundary and
  inflate the score. Phylogeny-aware validation is the honest test, and tightening that is
  on the roadmap. I'd rather state this plainly than quote a number I haven't fully earned.

If those caveats make the project look less finished, good — that's the accurate picture,
and I'd rather a reviewer trust my numbers than be dazzled by them.

## How it's built (the engineering shape)

```
genome/    -> runs AMRFinderPlus, parses gene + mutation calls
features/  -> turns calls into model features (gene presence, mutation counts, ESM-2 embeddings)
ml/        -> XGBoost training + per-antibiotic MIC models
explain/   -> SHAP attribution + CLSI breakpoints -> clinical report
data/      -> reproducible NCBI/NARMS acquisition + harmonization pipeline
cli/       -> typer-based command surface (predict / train / data)
```

It's a clean layered dependency flow — each layer only knows about the one below it — which
is what makes the AMRFinderPlus boundary and the "predict a number, then categorize" design
enforceable rather than aspirational. Packaged with `pyproject.toml`, a test suite (42
tests), and a data pipeline that reproduces the training set from public NCBI sources
without manual genome wrangling. It's `Development Status :: Alpha`, and labeled honestly as
such.

## The forward-looking bet

The piece I'm most curious about is using a protein language model (ESM-2) to represent
resistance-gene *variants* by their actual sequence, so the model can reason about a mutation
it has never seen rather than treating genes as on/off switches. It's wired in but not yet
beating the simpler baseline — which is its own honest result, and exactly the kind of
open question I find worth chasing.

---

*Built by [kglazier](https://github.com/kglazier). If you want the technical entry point,
start with the [README](README.md).*
