import os
import pickle
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pathlib import Path
import streamlit as st
import requests

load_dotenv()
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'gpt-4o-mini')
TOP_K = int(os.getenv('TOP_K', '4'))
EMB_MODEL = os.getenv('EMB_MODEL', 'all-MiniLM-L6-v2')
INDEX_PATH = os.getenv('INDEX_PATH', 'faiss_index')

st.set_page_config(page_title='RAG — LangChain-free demo')
st.title('RAG Q&A — Minimal RAG Demo')

st.markdown('---')
st.header('Upload PDFs to add to the knowledge base')
uploaded_files = st.file_uploader('Upload one or more PDF files', type=['pdf'], accept_multiple_files=True)
if uploaded_files:
    total_size = sum(f.size for f in uploaded_files)
    max_recommended = 50 * 1024 * 1024
    for uf in uploaded_files:
        size_mb = uf.size / (1024*1024)
        warn = ' (large)' if uf.size > max_recommended else ''
        st.write(f"- {uf.name} — {size_mb:.1f} MB{warn}")
    if total_size > max_recommended and not st.checkbox('Large upload, proceed anyway'):
        st.stop()
    if st.button('Save files & Ingest Index'):
        os.makedirs('data', exist_ok=True)
        saved = []
        for uf in uploaded_files:
            save_path = os.path.join('data', uf.name)
            base, ext = os.path.splitext(uf.name)
            c = 1
            while os.path.exists(save_path):
                save_path = os.path.join('data', f"{base}_{c}{ext}")
                c += 1
            with open(save_path, 'wb') as out:
                out.write(uf.getbuffer())
            saved.append(save_path)
        st.success(f"Saved {len(saved)} file(s). Starting ingest...")
        with st.spinner('Embedding & indexing PDFs...'):
            try:
                import subprocess, sys
                res = subprocess.run([sys.executable, 'ingest_and_index.py'], capture_output=True, text=True, timeout=1800)
                if res.returncode == 0:
                    st.success('Ingest completed successfully.')
                else:
                    st.error('Ingest failed.')
                    st.code(res.stdout + '\n' + res.stderr)
            except Exception as e:
                st.error(f'Error running ingest: {e}')
st.markdown('---')

# Ingest status display
status_path = 'faiss_index/status.json'
st.subheader('Ingest status')
if os.path.exists(status_path):
    try:
        with open(status_path, 'r') as sf:
            import json
            stdata = json.load(sf)
        st.write(stdata.get('message', 'No message'))
        total = stdata.get('total_chunks', 0)
        processed = stdata.get('processed_chunks', 0)
        if total > 0:
            st.progress(int(processed * 100 / total))
            st.write(f"Processed {processed}/{total} chunks.")
        st.write(f"Status: {stdata.get('status')}")
    except Exception as e:
        st.error(f'Error reading status: {e}')
else:
    st.write('No ingest in progress.')
st.markdown('---')



@st.cache_resource
def load_index():
    index = faiss.read_index(str(Path(INDEX_PATH) / 'index.faiss'))
    with open(Path(INDEX_PATH)/'docs.pkl', 'rb') as f:
        meta = pickle.load(f)
    emb_model = SentenceTransformer(EMB_MODEL)
    return index, meta['docs'], emb_model

index, docs, emb_model = load_index()

q = st.text_input('Ask a question about your docs')
if st.button('Search') and q.strip():
    try:
        q_emb = emb_model.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, TOP_K)
        retrieved = [docs[i] for i in I[0].tolist()]
        st.subheader('Retrieved context')
        for i, r in enumerate(retrieved):
            st.markdown(f'**Chunk {i+1}**')
            st.write(r[:800] + ('...' if len(r) > 800 else ''))
        # Call backend for answer (preferred path)
        resp = requests.post('http://localhost:8000/query', json={'question': q, 'top_k': TOP_K}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        answer = data.get('answer', '')
    except Exception as e:
        # Fallback direct Ollama if backend unavailable
        context = '\n\n'.join(retrieved) if 'retrieved' in locals() else ''
        prompt = f"Use the context below to answer the question. If the answer is not present, say 'I don't know.'\n\nContext:\n{context}\n\nQuestion: {q}\nAnswer:"
        try:
            resp2 = requests.post('http://localhost:11434/api/generate', json={'model': OLLAMA_MODEL, 'prompt': prompt, 'max_tokens': 512}, timeout=30)
            resp2.raise_for_status()
            data2 = resp2.json()
            answer = data2.get('choices', [{}])[0].get('message', {}).get('content', '') or data2.get('text', '')
        except Exception as e2:
            answer = f'Backend & fallback failed: {e}; {e2}'
    st.subheader('Answer')
    st.write(answer)