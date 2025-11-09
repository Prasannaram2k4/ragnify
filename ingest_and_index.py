import os
import glob
import pickle
import json
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
import numpy as np
import faiss
from tqdm import tqdm
import math
import time
import gc
from backend.services.embeddings import embed_texts

# Reduce default thread usage to limit memory/CPU on low-resource machines
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_DIR = Path(__file__).resolve().parent
load_dotenv()
EMB_MODEL = os.getenv('EMB_MODEL', os.getenv('HF_EMBED_MODEL', 'sentence-transformers/all-mpnet-base-v2'))
VECTOR_DIM = int(os.getenv('VECTOR_DIM', '768'))
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '400'))
OVERLAP = int(os.getenv('OVERLAP', '50'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '2'))  # very small default to reduce memory
STATUS_PATH = os.getenv('STATUS_PATH', str(BASE_DIR / 'faiss_index/status.json'))

model = None  # no local model used now (remote embeddings)

def hash_embed_batch(texts, dim: int = 1024, ngram: int = 3):
    """Low-memory hashing embedding: character n-gram counts into fixed dim.
    Returns numpy array of shape (len(texts), dim) L2-normalized.
    """
    import numpy as _np
    out = _np.zeros((len(texts), dim), dtype=_np.float32)
    for i, t in enumerate(texts):
        s = (t or "").lower()
        if len(s) < ngram:
            idx = hash(s) % dim
            out[i, idx] += 1.0
        else:
            for j in range(len(s) - ngram + 1):
                g = s[j:j+ngram]
                idx = (hash(g) & 0x7FFFFFFF) % dim
                out[i, idx] += 1.0
    # L2 normalize
    norms = _np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    out /= norms
    return out

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = []
    for p in reader.pages:
        text.append(p.extract_text() or "")
    return "\n".join(text)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    tokens = len(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return chunks


def write_status(obj):
    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    with open(STATUS_PATH, 'w') as f:
        json.dump(obj, f)


def ingest(input_pattern=None, index_path=None):
    if input_pattern is None:
        input_pattern = str(BASE_DIR / 'data' / '*.pdf')
    if index_path is None:
        index_path = str(BASE_DIR / 'faiss_index')
    # initialize status
    status = {'status': 'starting', 'total_chunks': 0, 'processed_chunks': 0, 'message': ''}
    write_status(status)

    docs = []
    ids = []
    file_paths = sorted(glob.glob(input_pattern))
    if not file_paths:
        status.update({'status': 'error', 'message': 'No files found matching pattern', 'total_chunks': 0})
        write_status(status)
        print(status['message'])
        return

    # Extract and chunk
    for path in file_paths:
        text = extract_text_from_pdf(path)
        for i, chunk in enumerate(chunk_text(text)):
            docs.append(chunk)
            ids.append(f"{Path(path).stem}_{i}")

    total = len(docs)
    status.update({'status': 'indexing', 'total_chunks': total, 'processed_chunks': 0, 'message': 'Starting embedding and indexing...'})
    write_status(status)

    # batch encode and add to FAISS incrementally
    dim = None
    index = None
    embeddings_acc = []
    global model
    try:
        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            batch = docs[start:end]
            embs = embed_texts(batch)
            if index is None:
                dim = embs.shape[1]
                index = faiss.IndexFlatIP(dim)
            index.add(embs)

            # write intermediate index to disk to avoid losing progress on crash
            os.makedirs(index_path, exist_ok=True)
            faiss.write_index(index, os.path.join(index_path, 'index.faiss'))

            # save docs for retrieval (append new docs)
            docs_path = os.path.join(index_path, 'docs.pkl')
            if os.path.exists(docs_path):
                # load existing and append
                with open(docs_path, 'rb') as f:
                    existing = pickle.load(f)
                existing_docs = existing.get('docs', [])
                existing_ids = existing.get('ids', [])
                new_docs = existing_docs + docs[start:end]
                new_ids = existing_ids + ids[start:end]
            else:
                new_docs = docs[:end]
                new_ids = ids[:end]
            with open(docs_path, 'wb') as f:
                pickle.dump({'docs': new_docs, 'ids': new_ids}, f)

            # update status
            status['processed_chunks'] = end
            status['message'] = f'Processed {end}/{total} chunks.'
            write_status(status)
            # small sleep to allow UI to pick up changes (optional)
            time.sleep(0.1)
            # Free memory aggressively
            del embs
            gc.collect()
            # Optionally unload model after each batch to further lower peak RAM
            # No local model to unload; still force GC
            gc.collect()

        status.update({'status': 'done', 'processed_chunks': total, 'message': 'Ingest completed successfully.'})
        write_status(status)
        print('Ingest completed:', total, 'chunks.')
    except Exception as e:
        status.update({'status': 'error', 'message': str(e)})
        write_status(status)
        raise

if __name__ == '__main__':
    ingest()
