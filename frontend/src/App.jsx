import React, { useEffect, useRef, useState } from 'react'
import './styles.css'

export default function App(){
  const [q, setQ] = useState('')
  const [ans, setAns] = useState('')
  const [loading, setLoading] = useState(false)
  const [retrieved, setRetrieved] = useState([])
  const [files, setFiles] = useState([])
  const [uploadStatus, setUploadStatus] = useState('')
  const [ingestInfo, setIngestInfo] = useState({status:'idle', processed_chunks:0, total_chunks:0, message:''})
  const pollRef = useRef(null)
  const [backendOk, setBackendOk] = useState(null) // null=unknown, true=ok, false=down
  const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
  const apiKey = import.meta.env.VITE_API_KEY || ''

  async function ask(){
    setLoading(true)
    setAns('')
    try{
      const res = await fetch(`${backendUrl}/query`, {
        method: 'POST',
        headers: {
          'Content-Type':'application/json',
          ...(apiKey ? {'X-API-KEY': apiKey} : {})
        },
        body: JSON.stringify({question: q, top_k: 4})
      })
      if(!res.ok){
        const err = await res.json()
        setAns('Error: ' + (err.detail || res.statusText))
        setLoading(false)
        return
      }
      const data = await res.json()
      setAns(data.answer || '(No answer returned — showing extractive fallback)')
      setRetrieved(data.retrieved || [])
    }catch(e){
      setAns('Network error: ' + e.message)
    }finally{
      setLoading(false)
    }
  }

  function onFileChange(e){
    setFiles(Array.from(e.target.files))
  }

  async function upload(){
    if(!files.length){
      setUploadStatus('Select PDF files first.')
      return
    }
    setUploadStatus('Uploading...')
    const form = new FormData()
    files.forEach(f => form.append('files', f))
    try{
      const res = await fetch(`${backendUrl}/upload`, {
        method: 'POST',
        headers: {
          ...(apiKey ? {'X-API-KEY': apiKey} : {})
        },
        body: form
      })
      const data = await res.json()
      if(!res.ok){
        setUploadStatus('Upload error: ' + (data.detail || res.statusText))
        return
      }
      setUploadStatus(`Uploaded: ${data.uploaded.join(', ')} (ingest started)`) 
      // begin polling ingest status
      startPollingIngest()
    }catch(e){
      setUploadStatus('Network error: ' + e.message)
    }
  }

  function startPollingIngest(){
    stopPollingIngest()
    pollRef.current = setInterval(async () => {
      try{
        const res = await fetch(`${backendUrl}/ingest_status`, {
          headers: {
            ...(apiKey ? {'X-API-KEY': apiKey} : {})
          }
        })
        if(!res.ok) return
        const data = await res.json()
        setIngestInfo(data)
        if(data.status === 'done' || data.status === 'error'){
          stopPollingIngest()
          setUploadStatus(data.status === 'done' ? 'Ingest completed. Index is ready.' : `Ingest error: ${data.message || ''}`)
        }
      }catch{
        // ignore transient errors
      }
    }, 1000)
  }

  function stopPollingIngest(){
    if(pollRef.current){
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  useEffect(()=>{
    // health ping
    const ping = async () => {
      try {
        const r = await fetch(`${backendUrl}/health`)
        setBackendOk(r.ok)
      } catch {
        setBackendOk(false)
      }
    }
    ping()
    const t = setInterval(ping, 5000)
    return () => { clearInterval(t); stopPollingIngest() }
  },[])

  return (
    <div className="container">
  <h1>RAG Q&amp;A {backendOk===true && <span className="badge online">backend: online</span>} {backendOk===false && <span className="badge offline">backend: down</span>}</h1>
      <textarea className="query" rows={4} value={q} onChange={e=>setQ(e.target.value)} placeholder="Ask a question about your docs" />
      <div className="controls">
        <button className="btn" onClick={ask} disabled={loading||!q}>Ask</button>
        {loading && <span className="loading">Loading…</span>}
      </div>
      <section className="upload">
        <h2>Upload PDFs</h2>
        <input type="file" accept="application/pdf" multiple onChange={onFileChange} />
        <button className="btn" onClick={upload} disabled={!files.length}>Upload & Ingest</button>
        {uploadStatus && <p className="status">{uploadStatus}</p>}
        {(ingestInfo?.total_chunks || 0) > 0 && (
          <div className="ingest-progress">
            <div className="bar"><span style={{width: `${Math.min(100, Math.round(100*(ingestInfo.processed_chunks||0)/Math.max(1,(ingestInfo.total_chunks||0))))}%`}} /></div>
            <div className="meta">
              <span>{ingestInfo.processed_chunks}/{ingestInfo.total_chunks}</span>
              <span className={`badge ${ingestInfo.status}`}>{ingestInfo.status}</span>
            </div>
            {ingestInfo.message && <div className="note">{ingestInfo.message}</div>}
          </div>
        )}
      </section>
      <section className="result">
        <h2>Answer</h2>
        <pre className="answer">{ans || 'No answer yet. Ask a question after ingest completes, or try again.'}</pre>
      </section>
      <section className="retrieved">
        <h2>Retrieved Context</h2>
        {retrieved.map((r,i)=> (
          <div key={i} className="chunk">
            <strong>Chunk {i+1}</strong>
            <p>{r.slice(0,600)}{r.length>600?'...':''}</p>
          </div>
        ))}
      </section>
    </div>
  )
}
