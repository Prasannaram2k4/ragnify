import os
import time
import requests
import numpy as np
from typing import List

class EmbeddingError(Exception):
    pass

def _hf_feature_extraction(texts: List[str], timeout: int = 60) -> np.ndarray:
    """Call HF Inference API to get embeddings for texts.
    Prefer the models endpoint; fallback to legacy pipeline feature-extraction path.
    """
    token = os.getenv('HF_API_TOKEN', '')
    if not token:
        raise EmbeddingError('Hugging Face API token not configured (HF_API_TOKEN).')
    model = os.getenv('HF_EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    headers = {'Authorization': f'Bearer {token}'}

    attempts = [
        ('models', f'https://api-inference.huggingface.co/models/{model}', {'inputs': texts, 'options': {'wait_for_model': True}}),
        ('legacy', f'https://api-inference.huggingface.co/pipeline/feature-extraction/{model}', {'inputs': texts, 'options': {'wait_for_model': True}}),
    ]

    for name, url, body in attempts:
        backoff = 2
        for _ in range(3):
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                if not data:
                    return np.zeros((len(texts), 1), dtype=np.float32)
                # Data can be 2D (batch x dim) or 3D (batch x tokens x dim). Mean-pool if token-level.
                if isinstance(data[0][0], list):
                    pooled = []
                    for item in data:
                        arr = np.array(item, dtype=np.float32)
                        pooled.append(arr.mean(axis=0))
                    X = np.stack(pooled, axis=0)
                else:
                    X = np.array(data, dtype=np.float32)
                norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
                return X / norms
            if resp.status_code in (429, 503, 524):
                time.sleep(backoff)
                backoff = min(backoff * 2, 10)
                continue
            if resp.status_code in (404, 410):
                break
        # try next attempt
        continue
    raise EmbeddingError('HF embeddings failed across endpoints')


def embed_texts(texts: List[str]) -> np.ndarray:
    """Return L2-normalized embeddings for a list of texts using Hugging Face Inference API.
    """
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    # The endpoint supports batching. Keep batches small to be gentle.
    B = int(os.getenv('EMBED_BATCH', '8'))
    arrs = []
    for i in range(0, len(texts), B):
        chunk = texts[i:i+B]
        X = _hf_feature_extraction(chunk)
        arrs.append(X)
    return np.vstack(arrs)
