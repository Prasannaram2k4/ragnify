import React, { useEffect, useState } from 'react'
import './styles.css'

const API = import.meta.env.VITE_BACKEND_URL || '/api'
const API_KEY = import.meta.env.VITE_API_KEY || ''
const headers = API_KEY ? { 'X-API-KEY': API_KEY } : {}

async function readJson(res) {
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || res.statusText)
  return data
}

export default function App() {
  const [q, setQ] = useState('')
  const [ans, setAns] = useState('')
  const [provider, setProvider] = useState('')
  const [retrieved, setRetrieved] = useState([])
  const [loading, setLoading] = useState(false)
  const [files, setFiles] = useState([])
  const [docs, setDocs] = useState([]) // [{name, chunks}]
  const [chunks, setChunks] = useState([]) // [{source, text}] kept client-side (serverless backend is stateless)
  const [status, setStatus] = useState('')
  const [health, setHealth] = useState(null)

  useEffect(() => {
    fetch(`${API}/health`).then(readJson).then(setHealth).catch(() => setHealth(false))
  }, [])

  async function ask() {
    setLoading(true)
    setAns('')
    setRetrieved([])
    try {
      const res = await fetch(`${API}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...headers },
        body: JSON.stringify({ question: q, top_k: 4, chunks }),
      })
      const data = await readJson(res)
      setAns(data.answer)
      setProvider(data.provider)
      setRetrieved(data.retrieved || [])
    } catch (e) {
      setAns('Error: ' + e.message)
    } finally {
      setLoading(false)
    }
  }

  async function upload() {
    setStatus('Uploading and indexing...')
    const form = new FormData()
    files.forEach((f) => form.append('files', f))
    try {
      const data = await readJson(await fetch(`${API}/upload`, { method: 'POST', headers, body: form }))
      setDocs((d) => [...d, ...data.files])
      setChunks((c) => [...c, ...data.chunks])
      setFiles([])
      setStatus(`Indexed ${data.chunks.length} chunks from ${data.files.length} file(s).`)
    } catch (e) {
      setStatus('Upload error: ' + e.message)
    }
  }

  function clearDocs() {
    setDocs([])
    setChunks([])
    setStatus('')
  }

  return (
    <div className="container">
      <h1>
        Ragnify RAG Q&amp;A
        {health && <span className="badge online">backend: online ({health.provider})</span>}
        {health === false && <span className="badge offline">backend: down</span>}
      </h1>
      <p className="note">
        Ask about the built-in sample document, or upload your own PDF/TXT/MD files (up to 4 MB total).
      </p>
      <textarea className="query" rows={3} value={q} onChange={(e) => setQ(e.target.value)}
        placeholder="e.g. How does Ragnify choose which chunks to use?" />
      <div className="controls">
        <button className="btn" onClick={ask} disabled={loading || !q.trim()}>Ask</button>
        {loading && <span className="loading">Thinking…</span>}
      </div>

      <section className="upload">
        <h2>Documents</h2>
        <input type="file" accept=".pdf,.txt,.md" multiple
          onChange={(e) => setFiles(Array.from(e.target.files))} />
        <button className="btn" onClick={upload} disabled={!files.length}>Upload &amp; Index</button>
        {docs.length > 0 && <button className="btn" onClick={clearDocs}>Clear</button>}
        {status && <p className="status">{status}</p>}
        <ul>{docs.map((d, i) => <li key={i}>{d.name} — {d.chunks} chunks</li>)}</ul>
      </section>

      <section className="result">
        <h2>Answer {provider && <span className="badge">{provider}</span>}</h2>
        <pre className="answer">{ans || 'No answer yet.'}</pre>
      </section>
      <section className="retrieved">
        <h2>Retrieved Context</h2>
        {retrieved.map((r, i) => (
          <div key={i} className="chunk">
            <strong>{r.source}</strong> <span className="note">score {r.score}</span>
            <p>{r.text.slice(0, 600)}{r.text.length > 600 ? '...' : ''}</p>
          </div>
        ))}
      </section>
    </div>
  )
}
