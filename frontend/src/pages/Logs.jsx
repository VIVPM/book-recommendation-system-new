import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import './Logs.css'

// Dynamically set API URL based on where the frontend is hosted
const API = window.location.hostname === 'localhost'
    ? 'http://localhost:8000'
    : `http://${window.location.hostname}:8000`
const PAGE_SIZE = 10

export default function LogsPage() {
    const [logs, setLogs] = useState([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [expanded, setExpanded] = useState({})
    const [page, setPage] = useState(1)

    const fetchLogs = () => {
        setLoading(true)
        setError(null)
        fetch(`${API}/logs`)
            .then(r => r.json())
            .then(data => {
                const files = data.logs || []
                setLogs(files)
                setExpanded(files.length ? { 0: true } : {})
                setPage(1)
            })
            .catch(() => setError('Could not connect to API server. Is it running on port 8000?'))
            .finally(() => setLoading(false))
    }

    useEffect(() => { fetchLogs() }, [])

    const toggle = (i) => setExpanded(prev => ({ ...prev, [i]: !prev[i] }))

    // Pagination
    const totalPages = Math.ceil(logs.length / PAGE_SIZE)
    const pageLogs = logs.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

    return (
        <div className="logs-page">
            {/* ── Header ── */}
            <header className="logs-header">
                <div className="logs-header-left">
                    <Link to="/" className="back-btn">← Back</Link>
                    <h1>🗂️ Pipeline Logs</h1>
                </div>
                <button className="refresh-btn" onClick={fetchLogs} disabled={loading}>
                    {loading ? '⏳' : '🔄'} Refresh
                </button>
            </header>

            {/* ── Body ── */}
            <main className="logs-container">
                {loading && (
                    <div className="logs-state">
                        <div className="logs-spinner" />
                        <p>Loading log files...</p>
                    </div>
                )}

                {error && <div className="logs-state logs-state-error"><p>⚠️ {error}</p></div>}

                {!loading && !error && logs.length === 0 && (
                    <div className="logs-state"><p>📭 No log files found.</p></div>
                )}

                {!loading && !error && logs.length > 0 && (
                    <>
                        {/* Log file list */}
                        {pageLogs.map((log, i) => {
                            const globalIdx = (page - 1) * PAGE_SIZE + i
                            return (
                                <div key={globalIdx} className={`log-file ${expanded[globalIdx] ? 'log-file--open' : ''}`}>
                                    <button className="log-file-header" onClick={() => toggle(globalIdx)}>
                                        <span className="log-badge">{globalIdx === 0 ? 'latest' : `#${logs.length - globalIdx}`}</span>
                                        <span className="log-filename">📄 {log.filename}</span>
                                        <span className="log-count">{log.lines.length} lines</span>
                                        <span className="log-chevron">{expanded[globalIdx] ? '▲' : '▼'}</span>
                                    </button>

                                    {expanded[globalIdx] && (
                                        <div className="log-content">
                                            {log.lines.length === 0
                                                ? <p className="log-empty-file">Empty file</p>
                                                : log.lines.map((line, j) => {
                                                    const cls = line.includes('ERROR') ? 'log-line log-error-line'
                                                        : line.includes('WARNING') ? 'log-line log-warn-line'
                                                            : line.includes('INFO') ? 'log-line log-info-line'
                                                                : 'log-line'
                                                    return <p key={j} className={cls}>{line}</p>
                                                })
                                            }
                                        </div>
                                    )}
                                </div>
                            )
                        })}

                        {/* Pagination */}
                        {totalPages > 1 && (
                            <div className="pagination">
                                <button
                                    className="page-btn"
                                    onClick={() => setPage(p => Math.max(1, p - 1))}
                                    disabled={page === 1}
                                >← Prev</button>

                                {Array.from({ length: totalPages }, (_, i) => i + 1).map(p => (
                                    <button
                                        key={p}
                                        className={`page-btn ${p === page ? 'page-btn--active' : ''}`}
                                        onClick={() => setPage(p)}
                                    >{p}</button>
                                ))}

                                <button
                                    className="page-btn"
                                    onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                                    disabled={page === totalPages}
                                >Next →</button>
                            </div>
                        )}
                    </>
                )}
            </main>
        </div>
    )
}
