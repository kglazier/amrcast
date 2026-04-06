# Potential Early Testers & Collaborators

Researchers and groups who have published about needing better WGS-based MIC prediction tools. Reach out when AMRCast has preliminary results on real data.

> **NOTE:** Twitter handles need verification before outreach. All source papers should be checked for corresponding author emails. Last researched: 2026-04-05.

---

## Tier 1 — Most Likely to Engage

These people are actively working on this exact problem and have publicly stated the need.

### Danesh Moradigaravand
- **Affiliation:** KAUST (King Abdullah University of Science and Technology), previously Wellcome Sanger Institute
- **What they said:** Explicitly stated the need for better training datasets and models for MIC prediction
- **Source:** "Prediction of antimicrobial resistance in Escherichia coli from large-scale pan-genome data" — Nature Communications, ~2018-2019. Also published K. pneumoniae MIC regression.
- **Date:** 2018-2019
- **Contact:** KAUST faculty page; corresponding author email in the Nature Comms paper
- **Why reach out:** They did E. coli MIC regression — this is the closest prior work to what we're building

### Finlay Maguire
- **Affiliation:** Dalhousie University, Faculty of Computer Science
- **What they said:** Benchmarked AMR prediction tools and identified accuracy gaps. Built hAMRonization (tool for harmonizing AMR prediction outputs across tools).
- **Source:** Benchmark/review papers on AMR prediction from WGS; hAMRonization GitHub repo
- **Date:** ~2022-2024
- **Contact:** Twitter: @finloymaguire (verify). GitHub: fmaguire. Email from Dalhousie faculty page or paper corresponding author.
- **Why reach out:** Knows every gap in every tool — ideal for honest critical feedback

### Rick Stevens / Maulik Shukla
- **Affiliation:** University of Chicago / Argonne National Laboratory (BV-BRC/PATRIC team)
- **What they said:** Published foundational paper on predicting MIC from WGS using ML. Acknowledged accuracy limitations especially for Gram-negatives.
- **Source:** "A white-box machine learning approach for revealing antibiotic mechanisms of action" — Nature Communications, 2018. Also BV-BRC AMR prediction service documentation.
- **Date:** 2018, ongoing BV-BRC work
- **Contact:** Argonne National Laboratory page (Stevens); BV-BRC team contact via bv-brc.org
- **Why reach out:** They host the data we train on (BV-BRC). Natural collaboration.

### Kat Holt / Kelly Wyres
- **Affiliation:** London School of Hygiene & Tropical Medicine (Holt, moved from Monash) / Monash University (Wyres)
- **What they said:** "Incrementing the population database is pivotal for future clinical implementation" — in context of Klebsiella MIC prediction work
- **Source:** Klebsiella genomics papers; Kleborate tool. Associated with the Microbial Genomics 2024 Kpneu MIC prediction paper.
- **Date:** ~2024
- **Contact:** Twitter: @DrKatHolt, @kellywyres. LSHTM / Monash faculty pages.
- **Why reach out:** Klebsiella MIC prediction — would want to see E. coli and multi-species expansion

### Zamin Iqbal
- **Affiliation:** EMBL-EBI (European Bioinformatics Institute), previously University of Oxford
- **What they said:** Published on the challenge of quantitative MIC prediction vs binary classification. Has discussed limitations of catalog-based approaches for MIC inference.
- **Source:** Mykrobe tool publications; commentary papers on WGS-based AMR
- **Date:** ~2019-2024
- **Contact:** Twitter: @ZaminIqbal (verify). EMBL-EBI staff page.
- **Why reach out:** Methodology expert on exactly the quantitative prediction problem

### Frank Aarestrup / Rene Hendriksen
- **Affiliation:** Technical University of Denmark (DTU), National Food Institute. WHO Collaborating Centre for Antimicrobial Resistance and Genomics.
- **What they said:** Has explicitly called for moving beyond binary R/S to quantitative MIC prediction from WGS
- **Source:** ResFinder publications; multiple review papers on WGS-based AMR surveillance
- **Date:** ~2017-2024 (ongoing advocacy)
- **Contact:** Twitter: @frankaarestrup (verify). DTU faculty page.
- **Why reach out:** Godfather of WGS-based AMR surveillance. Endorsement/feedback from him carries weight in the field.

---

## Tier 2 — Clinical Validation Partners

Could provide real-world clinical testing and validation.

### Day Zero Diagnostics
- **Affiliation:** Boston, MA. CEO: Miriam Huntley. CSO: Jason Withers.
- **What they said:** Building Keynome platform for rapid WGS-based diagnostics. MicrohmDB database (48K+ genomes, 450K+ AST results). Currently predict R/S categories (>90% categorical agreement), NOT quantitative MIC.
- **Source:** dayzerodiagnostics.com/technology; IDWeek conference presentations; PMC article PMC10678852
- **Date:** Ongoing as of 2025
- **Contact:** Company website. Miriam Huntley on LinkedIn.
- **Why reach out:** Commercial partner potential. Quantitative MIC extends their clinical workflow directly.

### UKHSA Genomics Team
- **Affiliation:** UK Health Security Agency (formerly PHE), Colindale. Key people: Silke Sheridan, Katie Hopkins, Theresa Lamagni.
- **What they said:** Invested in moving from phenotypic AST to WGS-inferred resistance for E. coli bloodstream infection surveillance
- **Source:** UKHSA publications on E. coli BSI genomic surveillance
- **Date:** ~2022-2025
- **Contact:** UKHSA institutional contact; corresponding authors on UKHSA surveillance publications
- **Why reach out:** Large-scale E. coli WGS surveillance — exactly our target species

### Amy Mathers
- **Affiliation:** University of Virginia, Division of Infectious Diseases
- **What they said:** Highlighted the gap between WGS capability and clinical utility for MIC prediction in Gram-negatives
- **Source:** Publications on WGS for clinical AMR in Gram-negatives
- **Date:** ~2020-2024
- **Contact:** UVA faculty page
- **Why reach out:** Clinical microbiologist perspective — can validate clinical utility

### CDC AR Lab Network
- **Affiliation:** CDC Division of Healthcare Quality Promotion. Key contact: Alison Laufer Halpin.
- **What they said:** Need standardized tools to interpret WGS for resistance
- **Source:** CDC AR Lab Network publications and reports
- **Date:** Ongoing
- **Contact:** CDC AR Lab Network website; corresponding authors on CDC AMR publications
- **Why reach out:** National surveillance network — if they adopt it, massive impact

---

## Tier 3 — Methodology Collaborators

Could help improve the ML/bioinformatics approach.

### John Lees
- **Affiliation:** EMBL-EBI (European Bioinformatics Institute), previously Imperial College
- **What they said:** Developer of pyseer (bacterial GWAS tool). Co-author on GeneBac (deep learning for bacterial phenotype prediction including MIC).
- **Source:** pyseer documentation; GeneBac bioRxiv preprint 2024 (doi: 10.1101/2024.01.03.574022)
- **Date:** 2024
- **Contact:** Twitter: @johnlees6 (verify). GitHub: johnlees. EMBL-EBI staff page.

### Andrew McArthur
- **Affiliation:** McMaster University, Hamilton, Ontario
- **What they said:** Maintains CARD/RGI. Has discussed extending to quantitative prediction beyond gene detection.
- **Source:** CARD publications; conference talks
- **Date:** Ongoing
- **Contact:** Twitter: @ahmmcarthur. McMaster faculty page. CARD website: card.mcmaster.ca

### Nick Loman
- **Affiliation:** University of Birmingham, Institute of Microbiology and Infection
- **What they said:** Pioneer of nanopore sequencing for clinical microbiology. Consistently discusses gaps in clinical bioinformatics tools.
- **Source:** Multiple publications and Twitter commentary
- **Date:** Ongoing
- **Contact:** Twitter: @pathgenomics. Birmingham faculty page.

### Tim Read
- **Affiliation:** Emory University
- **What they said:** Written commentary on the gap between WGS promise and clinical deployment for AMR
- **Source:** Review/commentary papers on WGS clinical microbiology
- **Date:** ~2020-2024
- **Contact:** Twitter: @timtread (verify). Emory faculty page.

---

## Key Source Papers to Pull Contact Info From

These papers have corresponding author emails and are directly relevant:

1. Moradigaravand et al. "Prediction of antibiotic resistance in Escherichia coli from large-scale pan-genome data" — Nature Communications, ~2018
2. Nguyen et al. "Using Machine Learning to Predict Antimicrobial MICs and Associated Genomic Features for Nontyphoidal Salmonella" — JCM, 2019
3. Kpneu MIC prediction paper — Microbial Genomics, 2024 (search: "Klebsiella pneumoniae MIC prediction optimization")
4. GeneBac preprint — bioRxiv, 2024 (doi: 10.1101/2024.01.03.574022)
5. AMR benchmark paper — Briefings in Bioinformatics, 2024 (doi: 10.1093/bib/bbae206)
6. "Open problems in bacterial genomics" — Bioinformatics, 2025 (doi: 10.1093/bioinformatics/btaf206)
7. CRyPTIC data compendium — PLOS Biology (doi: 10.1371/journal.pbio.3001721)

---

## Outreach Strategy

When we have a working model with real E. coli MIC predictions (even preliminary):

1. **Open source the tool first** — credibility matters in this community
2. **Lead with their own words** — "You said X was needed, we built it"
3. **Ask for feedback, not endorsement** — researchers engage more when asked to critique
4. **Offer to run their data** — lower friction than asking them to install something
5. **Target a conference poster/preprint** — ASM Microbe, ECCMID, or bioRxiv gives legitimacy before peer review
