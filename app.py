"""Gradio entrypoint for Hugging Face Spaces (free Gradio SDK). Reuses the same RAG services as the FastAPI backend."""
import os
from pathlib import Path

import gradio as gr

from backend.services.llm_providers import generate, resolve_provider
from backend.services.rag import chunk_text, extract_text, load_sample_chunks, retrieve, search_persisted

MAX_BYTES = int(os.getenv('MAX_UPLOAD_BYTES', str(20 * 1024 * 1024)))


def index_files(paths, chunks):
    chunks = list(chunks or [])
    notes = []
    for p in paths or []:
        path = Path(p)
        data = path.read_bytes()
        if len(data) > MAX_BYTES:
            notes.append(f'{path.name}: too large, skipped')
            continue
        try:
            pieces = chunk_text(extract_text(path.name, data))
        except Exception as e:
            notes.append(f'{path.name}: could not read ({e})')
            continue
        chunks.extend({'source': path.name, 'text': c} for c in pieces)
        notes.append(f'{path.name}: {len(pieces)} chunks')
    return chunks, '\n'.join(notes) or 'No files selected.'


def ask(question, chunks, top_k):
    question = (question or '').strip()
    if not question:
        return 'Enter a question.', ''
    corpus = load_sample_chunks() + list(chunks or [])
    top_k = int(top_k)
    hits = sorted(search_persisted(question, top_k) + retrieve(question, corpus, top_k),
                  key=lambda r: r['score'], reverse=True)[:top_k]
    answer, provider = generate(question, [h['text'] for h in hits])
    context = '\n\n'.join(f"**{h['source']}** (score {h['score']})\n{h['text'][:600]}" for h in hits)
    return f'{answer}\n\n_(answered by: {provider})_', context


with gr.Blocks(title='Ragnify') as demo:
    gr.Markdown(f'# Ragnify RAG Q&A\nAsk about the built-in sample, or upload PDF/TXT/MD files. LLM provider: `{resolve_provider()}`')
    state = gr.State([])
    with gr.Row():
        files = gr.File(file_count='multiple', file_types=['.pdf', '.txt', '.md'], label='Documents')
        status = gr.Textbox(label='Index status', lines=4, interactive=False)
    gr.Button('Upload & Index').click(index_files, [files, state], [state, status])
    q = gr.Textbox(label='Question', lines=2)
    k = gr.Slider(1, 10, value=4, step=1, label='Top-k chunks')
    ans = gr.Markdown(label='Answer')
    ctx = gr.Markdown(label='Retrieved context')
    gr.Button('Ask', variant='primary').click(ask, [q, state, k], [ans, ctx])

if __name__ == '__main__':
    demo.launch()
