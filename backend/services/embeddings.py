import os
import re
import zlib
from typing import List

import numpy as np
import requests

HASH_DIM = 2048
HF_MAX_TEXTS = int(os.getenv('HF_EMBED_MAX_TEXTS', '128'))
_TOKEN_RE = re.compile(r'[a-z0-9]+')


class EmbeddingError(Exception):
    pass


def _bucket(s: str) -> int:
    return zlib.crc32(s.encode('utf-8')) % HASH_DIM


def _normalize(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def hash_embed(texts: List[str]) -> np.ndarray:
    """Deterministic, dependency-free embedding: hashed word unigrams/bigrams plus char trigrams."""
    out = np.zeros((len(texts), HASH_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        words = _TOKEN_RE.findall((t or '').lower())
        for w in words:
            out[i, _bucket('w:' + w)] += 1.0
            padded = f'#{w}#'
            for j in range(len(padded) - 2):
                out[i, _bucket('c:' + padded[j:j + 3])] += 0.2
        for a, b in zip(words, words[1:]):
            out[i, _bucket('b:' + a + ' ' + b)] += 0.7
    return _normalize(np.log1p(out))


def _hf_embed(texts: List[str], timeout: int = 30) -> np.ndarray:
    token = os.getenv('HF_API_TOKEN', '')
    if not token:
        raise EmbeddingError('HF_API_TOKEN not configured')
    model = os.getenv('HF_EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    url = f'https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction'
    batch = 32
    parts = []
    for i in range(0, len(texts), batch):
        sub = texts[i:i + batch]
        try:
            resp = requests.post(
                url,
                headers={'Authorization': f'Bearer {token}'},
                json={'inputs': sub, 'options': {'wait_for_model': True}},
                timeout=timeout,
            )
            resp.raise_for_status()
            X = np.array(resp.json(), dtype=np.float32)
        except Exception as e:
            raise EmbeddingError(f'HF embeddings failed: {e}') from e
        if X.ndim == 3:  # token-level output -> mean pool
            X = X.mean(axis=1)
        if X.ndim != 2 or X.shape[0] != len(sub):
            raise EmbeddingError('Unexpected HF embedding response shape')
        parts.append(X)
    return _normalize(np.vstack(parts))


def embed_texts(texts: List[str]) -> np.ndarray:
    """L2-normalized embeddings. Uses Hugging Face when configured, otherwise (or on failure) hashing.
    All texts in one call share the same backend so they are comparable."""
    if not texts:
        return np.zeros((0, HASH_DIM), dtype=np.float32)
    if (os.getenv('EMB_BACKEND', 'auto').lower() != 'hash'
            and os.getenv('HF_API_TOKEN') and len(texts) <= HF_MAX_TEXTS):
        try:
            return _hf_embed(texts)
        except EmbeddingError:
            pass
    return hash_embed(texts)
