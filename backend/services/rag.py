import io
import os
import pickle
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from pypdf import PdfReader

from .embeddings import embed_texts

SAMPLE_DIR = Path(__file__).resolve().parent.parent / 'sample_docs'


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    text = ' '.join(text.split())
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith('.pdf'):
        reader = PdfReader(io.BytesIO(data))
        return '\n'.join(p.extract_text() or '' for p in reader.pages)
    return data.decode('utf-8', errors='ignore')


def load_sample_chunks() -> List[Dict[str, str]]:
    out = []
    if SAMPLE_DIR.exists():
        for p in sorted(SAMPLE_DIR.glob('*.txt')):
            out.extend({'source': p.name, 'text': c} for c in chunk_text(p.read_text(encoding='utf-8')))
    return out


def retrieve(question: str, chunks: List[Dict[str, str]], top_k: int) -> List[Dict]:
    """Build an in-memory FAISS inner-product index (cosine on normalized vectors) and search it."""
    if not chunks:
        return []
    X = embed_texts([question] + [c['text'] for c in chunks])
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(np.ascontiguousarray(X[1:]))
    D, I = index.search(np.ascontiguousarray(X[:1]), min(top_k, len(chunks)))
    return [{'source': chunks[i]['source'], 'text': chunks[i]['text'], 'score': round(float(d), 4)}
            for d, i in zip(D[0], I[0]) if i >= 0]


def search_persisted(question: str, top_k: int) -> List[Dict]:
    """Search the on-disk index built by ingest_and_index.py (local/Docker); empty if absent or incompatible."""
    path = Path(os.getenv('INDEX_PATH', 'faiss_index'))
    if not (path / 'index.faiss').exists() or not (path / 'docs.pkl').exists():
        return []
    try:
        index = faiss.read_index(str(path / 'index.faiss'))
        with open(path / 'docs.pkl', 'rb') as f:
            docs = pickle.load(f).get('docs', [])
        q = embed_texts([question])
        if q.shape[1] != index.d:
            return []
        D, I = index.search(q, min(top_k, index.ntotal))
        return [{'source': 'faiss_index', 'text': docs[i], 'score': round(float(d), 4)}
                for d, i in zip(D[0], I[0]) if 0 <= i < len(docs)]
    except Exception:
        return []
