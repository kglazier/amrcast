"""ESM-2 protein language model embeddings for AMR proteins.

Uses Meta's ESM-2 (650M parameter model) as a frozen feature extractor.
Each AMR protein detected by AMRFinderPlus gets embedded into a 1280-dim
vector that captures structural and functional properties beyond what
sequence identity alone can provide.

This is the novel contribution of AMRCast — existing tools treat
"blaTEM-1 present" and "blaTEM-3 present" as equivalent binary features,
but they encode very different resistance profiles. ESM-2 embeddings
capture this functional difference.
"""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ESM-2 model configs: name -> embedding dimension
ESM_MODELS = {
    "esm2_t6_8M_UR50D": 320,       # tiny, fast, CPU-friendly
    "esm2_t12_35M_UR50D": 480,     # small
    "esm2_t30_150M_UR50D": 640,    # medium
    "esm2_t33_650M_UR50D": 1280,   # default — best accuracy/speed ratio
    "esm2_t36_3B_UR50D": 2560,     # large, needs >16GB VRAM
}


def _get_device():
    """Get the best available device."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_esm_model(model_name: str = "esm2_t33_650M_UR50D"):
    """Load ESM-2 model and alphabet.

    Returns (model, alphabet, embedding_dim).
    """
    import esm

    logger.info(f"Loading ESM-2 model: {model_name}")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.eval()

    embedding_dim = ESM_MODELS.get(model_name, 1280)
    device = _get_device()
    model = model.to(device)
    logger.info(f"ESM-2 loaded on {device} (embedding dim: {embedding_dim})")

    return model, alphabet, embedding_dim


def extract_protein_embedding(
    sequence: str,
    model,
    alphabet,
    device=None,
) -> np.ndarray:
    """Extract mean-pooled embedding for a single protein sequence.

    Args:
        sequence: Amino acid sequence string.
        model: Loaded ESM-2 model.
        alphabet: ESM-2 alphabet.
        device: Torch device.

    Returns:
        1D numpy array of shape (embedding_dim,).
    """
    import torch

    if device is None:
        device = next(model.parameters()).device

    # Truncate very long proteins to avoid OOM (ESM-2 max is 1022 tokens)
    max_len = 1022
    seq = sequence[:max_len]

    batch_converter = alphabet.get_batch_converter()
    data = [("protein", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        results = model(tokens, repr_layers=[model.num_layers])

    # Mean pool over sequence length (exclude BOS/EOS tokens)
    embeddings = results["representations"][model.num_layers]
    seq_embedding = embeddings[0, 1:len(seq)+1, :].mean(dim=0)

    return seq_embedding.cpu().numpy()


def extract_embeddings_batch(
    sequences: list[tuple[str, str]],
    model,
    alphabet,
    device=None,
    max_batch_tokens: int = 4096,
) -> dict[str, np.ndarray]:
    """Extract embeddings for multiple proteins efficiently.

    Batches proteins by similar length to maximize GPU utilization.

    Args:
        sequences: List of (label, sequence) tuples.
        model: Loaded ESM-2 model.
        alphabet: ESM-2 alphabet.
        device: Torch device.
        max_batch_tokens: Maximum total tokens per batch (controls GPU memory).

    Returns:
        Dict of label -> embedding array.
    """
    import torch

    if device is None:
        device = next(model.parameters()).device

    batch_converter = alphabet.get_batch_converter()
    max_len = 1022

    # Truncate and sort by length for efficient batching
    truncated = [(label, seq[:max_len]) for label, seq in sequences]
    truncated.sort(key=lambda x: len(x[1]))

    embeddings = {}

    # Process in batches
    batch = []
    batch_tokens_count = 0

    def _process_batch(batch_data):
        if not batch_data:
            return
        labels_batch, _, tokens = batch_converter(batch_data)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = model(tokens, repr_layers=[model.num_layers])

        reps = results["representations"][model.num_layers]
        for i, (label, seq) in enumerate(batch_data):
            seq_emb = reps[i, 1:len(seq)+1, :].mean(dim=0)
            embeddings[label] = seq_emb.cpu().numpy()

    for label, seq in truncated:
        seq_tokens = len(seq) + 2  # +2 for BOS/EOS
        if batch and batch_tokens_count + seq_tokens > max_batch_tokens:
            _process_batch(batch)
            batch = []
            batch_tokens_count = 0

        batch.append((label, seq))
        batch_tokens_count += seq_tokens

    _process_batch(batch)

    logger.info(f"Extracted embeddings for {len(embeddings)} proteins")
    return embeddings


class ESMEmbedder:
    """Manages ESM-2 model loading, embedding extraction, and caching."""

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        cache_dir: Path | None = None,
    ):
        self.model_name = model_name
        self.embedding_dim = ESM_MODELS.get(model_name, 1280)
        self.cache_dir = cache_dir
        self._model = None
        self._alphabet = None

    def _ensure_loaded(self):
        if self._model is None:
            self._model, self._alphabet, _ = _load_esm_model(self.model_name)

    def _cache_key(self, sequence: str) -> str:
        """Generate a cache key from sequence hash."""
        return hashlib.md5(sequence.encode()).hexdigest()

    def _load_cached(self, cache_key: str) -> np.ndarray | None:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{cache_key}.npy"
        if path.exists():
            return np.load(path)
        return None

    def _save_cached(self, cache_key: str, embedding: np.ndarray):
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.cache_dir / f"{cache_key}.npy", embedding)

    def embed_proteins(
        self,
        proteins: list[tuple[str, str]],
    ) -> dict[str, np.ndarray]:
        """Embed a list of proteins, using cache where available.

        Args:
            proteins: List of (label, sequence) tuples.

        Returns:
            Dict of label -> embedding numpy array (shape: embedding_dim).
        """
        results = {}
        to_compute = []

        # Check cache first
        for label, seq in proteins:
            key = self._cache_key(seq)
            cached = self._load_cached(key)
            if cached is not None:
                results[label] = cached
            else:
                to_compute.append((label, seq))

        if to_compute:
            logger.info(
                f"Computing ESM-2 embeddings: {len(to_compute)} new, "
                f"{len(results)} cached"
            )
            self._ensure_loaded()
            new_embeddings = extract_embeddings_batch(
                to_compute, self._model, self._alphabet
            )
            for label, seq in to_compute:
                if label in new_embeddings:
                    emb = new_embeddings[label]
                    results[label] = emb
                    self._save_cached(self._cache_key(seq), emb)
        else:
            logger.info(f"All {len(results)} embeddings loaded from cache")

        return results

    def embed_genome_proteins(
        self,
        gene_symbols: list[str],
        protein_sequences: dict[str, str],
    ) -> np.ndarray:
        """Embed AMR proteins for a genome and aggregate into fixed-length vector.

        Uses mean pooling over all AMR protein embeddings to produce a single
        genome-level embedding. Future versions could use attention-weighted
        pooling or a set transformer.

        Args:
            gene_symbols: List of gene symbols to embed.
            protein_sequences: Dict of gene_symbol -> protein sequence.

        Returns:
            1D numpy array of shape (embedding_dim,). Zero vector if no proteins.
        """
        proteins = [
            (sym, protein_sequences[sym])
            for sym in gene_symbols
            if sym in protein_sequences and protein_sequences[sym]
        ]

        if not proteins:
            return np.zeros(self.embedding_dim)

        embeddings = self.embed_proteins(proteins)

        if not embeddings:
            return np.zeros(self.embedding_dim)

        # Mean pool across all AMR protein embeddings
        emb_matrix = np.stack(list(embeddings.values()))
        return emb_matrix.mean(axis=0)
