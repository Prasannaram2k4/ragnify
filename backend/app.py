import logging
import os
from typing import List

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

from .services.llm_providers import generate, resolve_provider
from .services.rag import chunk_text, extract_text, load_sample_chunks, retrieve, search_persisted

logger = logging.getLogger('rag-backend')

MAX_UPLOAD_BYTES = int(os.getenv('MAX_UPLOAD_BYTES', str(4 * 1024 * 1024)))  # Vercel body limit is ~4.5MB
MAX_CHUNKS = int(os.getenv('MAX_CHUNKS', '1500'))

app = FastAPI(title='Ragnify RAG Backend')
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=os.getenv('CORS_ORIGIN_REGEX', r'http://(localhost|127\.0\.0\.1):\d+'),
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

api_key_header = APIKeyHeader(name='X-API-KEY', auto_error=False)


def check_api_key(api_key: str = Depends(api_key_header)):
    expected = os.getenv('API_KEY', '')
    if expected and api_key != expected:
        raise HTTPException(status_code=401, detail='Invalid or missing API key')
    return True


class Chunk(BaseModel):
    source: str
    text: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(4, ge=1, le=10)
    chunks: List[Chunk] = []  # user-uploaded chunks; the serverless backend is stateless
    use_samples: bool = True


router = APIRouter()


@router.get('/health')
def health():
    return {'status': 'ok', 'provider': resolve_provider(), 'sample_chunks': len(load_sample_chunks())}


@router.post('/upload')
async def upload(files: List[UploadFile] = File(...), ok: bool = Depends(check_api_key)):
    chunks, summary, total = [], [], 0
    for uf in files:
        name = os.path.basename(uf.filename or 'unnamed')
        if not name.lower().endswith(('.pdf', '.txt', '.md')):
            raise HTTPException(status_code=400, detail=f'{name}: only PDF, TXT and MD files are supported')
        data = await uf.read()
        total += len(data)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f'Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit')
        try:
            text = extract_text(name, data)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f'{name}: could not read file ({e})')
        pieces = chunk_text(text)
        if not pieces:
            raise HTTPException(status_code=422, detail=f'{name}: no extractable text')
        chunks.extend({'source': name, 'text': c} for c in pieces)
        summary.append({'name': name, 'chunks': len(pieces)})
    return {'files': summary, 'chunks': chunks}


@router.post('/query')
def query(req: QueryRequest, ok: bool = Depends(check_api_key)):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail='Empty question')
    corpus = [c.model_dump() for c in req.chunks[:MAX_CHUNKS]]
    if req.use_samples or not corpus:
        corpus = load_sample_chunks() + corpus
    persisted = search_persisted(q, req.top_k)
    if not corpus and not persisted:
        raise HTTPException(status_code=503, detail='No documents available. Upload a document first.')
    retrieved = sorted(persisted + retrieve(q, corpus, req.top_k), key=lambda r: r['score'], reverse=True)[:req.top_k]
    answer, provider = generate(q, [r['text'] for r in retrieved])
    return {'answer': answer, 'provider': provider, 'retrieved': retrieved}


# Served at both /api/* (Vercel) and /* (local dev, tests).
app.include_router(router)
app.include_router(router, prefix='/api')
