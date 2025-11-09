import os
import json
import pickle
import logging
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv, dotenv_values
import faiss
import requests
import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
# Ensure relative import works when running "uvicorn backend.app:app" from project root or backend dir
try:
    from services.llm_providers import call_ollama, call_openai, call_anthropic, call_huggingface  # type: ignore
    from services.embeddings import embed_texts, EmbeddingError  # type: ignore
except ModuleNotFoundError:
    # Fallback to explicit relative import
    from .services.llm_providers import call_ollama, call_openai, call_anthropic, call_huggingface  # type: ignore
    from .services.embeddings import embed_texts, EmbeddingError  # type: ignore

# Always load env from this backend folder, and also try project root .env
backend_env = Path(__file__).resolve().parent / '.env'
root_env = Path(__file__).resolve().parent.parent / '.env'
# Explicitly load backend first then root to allow root overrides if desired
if backend_env.exists():
    load_dotenv(dotenv_path=backend_env, override=True)
if root_env.exists():
    load_dotenv(dotenv_path=root_env, override=False)
INDEX_PATH = os.getenv('INDEX_PATH', 'faiss_index')
EMB_MODEL = os.getenv('EMB_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
EMB_BACKEND = os.getenv('EMB_BACKEND', os.getenv('USE_HASH_EMB', 'st')).lower()  # 'st' or 'hash'
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'huggingface')  # default to Hugging Face per project config
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'gpt-4o-mini')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
API_KEY = os.getenv('API_KEY', '')  # Optional simple auth token

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('rag-backend')

app = FastAPI(title='RAG Backend')

# Allow browser dev servers / local hosts to access
app.add_middleware(
    CORSMiddleware,
    # Use a regex to allow any localhost/127.0.0.1 port during dev (Vite may hop ports: 5173, 5174, 5175, etc.)
    # This avoids Disallowed CORS origin errors when the dev server picks a new port.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Simple API key header dependency (optional)
api_key_header = APIKeyHeader(name='X-API-KEY', auto_error=False)

def check_api_key(api_key: str = Depends(api_key_header)):
    if API_KEY:
        if not api_key or api_key != API_KEY:
            raise HTTPException(status_code=401, detail='Invalid or missing API key')
    return True

class QueryRequest(BaseModel):
    question: str
    top_k: int = 4

DATA_DIR = os.getenv('DATA_DIR', str(Path(__file__).resolve().parent.parent / 'data'))
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '800'))
OVERLAP = int(os.getenv('OVERLAP', '200'))
STATUS_PATH = os.getenv('STATUS_PATH', str(Path(__file__).resolve().parent.parent / 'faiss_index' / 'status.json'))

emb_model = None  # not used when using HF embeddings


@app.on_event('startup')
def load_resources():
    """Load FAISS index and embedding model on startup.
    Do not crash if the index isn't built yet; allow /health and /upload to work.
    """
    global index, docs, emb_model
    try:
        idx_path = Path(INDEX_PATH) / 'index.faiss'
        docs_path = Path(INDEX_PATH) / 'docs.pkl'
        if not idx_path.exists() or not docs_path.exists():
            index = None
            docs = []
            # don't load emb_model yet; defer to first query
            logger.warning('FAISS index not found yet. Build it by uploading PDFs or running ingest_and_index.py')
            return
        index = faiss.read_index(str(idx_path))
        with open(docs_path, 'rb') as f:
            meta = pickle.load(f)
        docs = meta.get('docs', [])
    # defer emb_model load to first query to reduce startup memory
        logger.info('Loaded FAISS index and embedding model.')
    except Exception as e:
        logger.error(f'Failed to initialize resources: {e}')
        # don't raise to allow server start; queries will guard on index presence

"""Provider calls are imported from services.llm_providers for cleaner structure."""

def call_llm(prompt):
    """Dispatch to selected LLM provider with built-in Hugging Face graceful fallback.
    If Hugging Face generation fails, return empty string so caller can degrade to extractive answer.
    """
    provider = LLM_PROVIDER.lower()
    try:
        if provider == 'ollama':
            return call_ollama(prompt)
        if provider == 'openai':
            return call_openai(prompt)
        if provider == 'anthropic':
            return call_anthropic(prompt)
        if provider == 'huggingface':
            ans = call_huggingface(prompt)
            return ans if isinstance(ans, str) else str(ans)
        raise HTTPException(status_code=500, detail=f'Unsupported LLM_PROVIDER: {LLM_PROVIDER}')
    except HTTPException as e:
        # Only suppress if huggingface provider; otherwise re-raise
        if provider == 'huggingface':
            logger.warning(f'Hugging Face generation error suppressed: {getattr(e, "detail", e)}')
            return ''
        raise

@app.get('/health')
def health():
    """Basic health check endpoint.
    Reports whether the FAISS index appears loaded."""
    ready = 'yes' if ('index' in globals() and index is not None and 'docs' in globals() and len(docs) > 0) else 'no'
    hf_embed = os.getenv('HF_EMBED_MODEL', EMB_MODEL)
    return {'status': 'ok', 'index_ready': ready, 'provider': LLM_PROVIDER, 'emb_model': hf_embed, 'version': 'v2'}

@app.post('/query')
def query(req: QueryRequest, ok: bool = Depends(check_api_key)):
    q = req.question
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail='Empty question')
    try:
        if index is None or not docs:
            # On-the-fly corpus from PDFs
            pdf_dir = Path(DATA_DIR)
            pdfs = list(pdf_dir.glob('*.pdf'))
            if not pdfs:
                raise HTTPException(status_code=503, detail='No index and no PDFs found in data/. Upload PDFs first.')
            try:
                from pypdf import PdfReader
            except Exception as e:
                raise HTTPException(status_code=500, detail=f'Missing PDF reader dependency: {e}')
            def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
                chunks = []
                start = 0
                n = len(text)
                while start < n:
                    end = min(start + chunk_size, n)
                    chunks.append(text[start:end])
                    if end == n:
                        break
                    start = max(0, end - overlap)
                return chunks
            corpus = []
            for p in pdfs:
                try:
                    reader = PdfReader(str(p))
                    t = "\n".join([pg.extract_text() or '' for pg in reader.pages])
                except Exception:
                    t = ''
                corpus.extend(chunk_text(t))
            if not corpus:
                raise HTTPException(status_code=503, detail='No text extracted from PDFs for retrieval.')
            # Retrieval with embedding fallback
            try:
                q_emb = embed_texts([q])
                sims = []
                B = int(os.getenv('EMBED_BATCH', '8'))
                for i in range(0, len(corpus), B):
                    batch = corpus[i:i+B]
                    X = embed_texts(batch)
                    s = (X @ q_emb.T).reshape(-1)
                    sims.extend([(float(s[j]), batch[j]) for j in range(len(batch))])
                sims.sort(key=lambda x: x[0], reverse=True)
                retrieved = [txt for _, txt in sims[:req.top_k]]
            except EmbeddingError:
                logger.warning('Embeddings unavailable; fallback to keyword retrieval.')
                q_terms = [t for t in q.lower().split() if t.isalpha() or t.isalnum()]
                def score(txt: str):
                    if not txt:
                        return 0.0
                    tokens = txt.lower().split()
                    hits = sum(1 for t in tokens if t in q_terms)
                    return hits / (len(tokens) + 1e-6)
                scored = sorted(((score(t), t) for t in corpus), key=lambda x: x[0], reverse=True)
                retrieved = [t for _, t in scored[:req.top_k]]
        else:
            try:
                q_emb = embed_texts([q])
                D, I = index.search(q_emb, req.top_k)
                retrieved = [docs[i] for i in I[0].tolist()]
            except EmbeddingError:
                logger.warning('Embeddings unavailable; fallback to keyword retrieval over docs.')
                q_terms = [t for t in q.lower().split() if t.isalpha() or t.isalnum()]
                def score(txt: str):
                    if not txt:
                        return 0.0
                    tokens = txt.lower().split()
                    hits = sum(1 for t in tokens if t in q_terms)
                    return hits / (len(tokens) + 1e-6)
                scored = sorted(((score(t), t) for t in docs), key=lambda x: x[0], reverse=True)
                retrieved = [t for _, t in scored[:req.top_k]]
        context = "\n\n".join(retrieved)
        prompt = f"Use the context below to answer the question. If the answer is not present, say 'I don't know.'\n\nContext:\n{context}\n\nQuestion: {q}\nAnswer:"
        try:
            answer = call_llm(prompt)
        except Exception as e:
            # Fallback: return extractive snippet when generation provider unavailable
            logger.warning(f'LLM generation failed ({getattr(e, "detail", str(e))}); returning extractive answer fallback.')
            snippet = (retrieved[0] if retrieved else '')[:600]
            answer = f"Based on the retrieved documents, here is a relevant excerpt:\n\n{snippet}"
        return {'answer': answer, 'retrieved': retrieved}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception('Query failed')
        raise HTTPException(status_code=500, detail=f'Query processing error: {e}')


def _run_ingest():
    """Run ingest process synchronously (called in background)."""
    try:
        # Add project root to path so we can import ingest_and_index when running from backend dir
        root_dir = Path(__file__).resolve().parent.parent
        if str(root_dir) not in os.sys.path:
            os.sys.path.append(str(root_dir))
        import ingest_and_index  # noqa: E401
        ingest_and_index.ingest()
        logger.info('Ingest after upload completed successfully.')
        # Reload FAISS index into memory so new queries use it immediately
        load_resources()
    except Exception as e:
        logger.error(f'Ingest after upload failed: {e}')


@app.post('/upload')
def upload_pdfs(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...), ok: bool = Depends(check_api_key)):
    """Upload one or more PDF files, store them, and trigger re-ingestion.
    Returns list of saved filenames and starts background ingest.
    """
    if not files:
        raise HTTPException(status_code=400, detail='No files provided')
    saved = []
    os.makedirs(DATA_DIR, exist_ok=True)
    for uf in files:
        fname = uf.filename or 'unnamed.pdf'
        if not fname.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f'File {fname} is not a PDF')
        # sanitize: remove path components
        fname = os.path.basename(fname)
        target = Path(DATA_DIR) / fname
        base, ext = os.path.splitext(fname)
        c = 1
        while target.exists():
            target = Path(DATA_DIR) / f"{base}_{c}{ext}"
            c += 1
        try:
            content = uf.file.read()
            with open(target, 'wb') as out:
                out.write(content)
            saved.append(target.name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Failed to save {fname}: {e}')

    # Trigger background ingest (rebuild index with new PDFs)
    background_tasks.add_task(_run_ingest)
    return {'uploaded': saved, 'ingest_started': True, 'data_dir': DATA_DIR}


@app.get('/ingest_status')
def ingest_status():
    """Return latest ingest status written by ingest_and_index.py.
    If no status file exists, report idle.
    """
    try:
        p = Path(STATUS_PATH)
        if not p.exists():
            return {'status': 'idle'}
        with open(p, 'r') as f:
            obj = json.load(f)
        # Minimal schema normalization
        obj.setdefault('status', 'unknown')
        obj.setdefault('processed_chunks', 0)
        obj.setdefault('total_chunks', 0)
        return obj
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to read ingest status: {e}')
