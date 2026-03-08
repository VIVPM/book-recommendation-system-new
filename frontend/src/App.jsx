import { useState, useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import './App.css'

const API = 'http://localhost:8000'
const DEFAULT_ZIP_URL = 'https://github.com/entbappy/Branching-tutorial/raw/master/books_data.zip'

// Training stages matching the backend pipeline
const TRAIN_STAGES = [
  { icon: '📥', label: 'Data Ingestion', detail: 'Downloading and extracting dataset...' },
  { icon: '🔍', label: 'Data Validation', detail: 'Cleaning and validating CSV files...' },
  { icon: '⚙️', label: 'Data Transformation', detail: 'Building book pivot table...' },
  { icon: '🤖', label: 'Model Training', detail: 'Training KNN model on book ratings...' },
]
// ── BookCard ──────────────────────────────────────────────────────────────────
function BookCard({ title, posterUrl, delay }) {
  const [imgError, setImgError] = useState(false)
  return (
    <div className="book-card" style={{ animationDelay: `${delay}s` }}>
      {posterUrl && !imgError ? (
        <img className="book-poster" src={posterUrl} alt={title} onError={() => setImgError(true)} />
      ) : (
        <div className="book-poster-placeholder">📚</div>
      )}
      <div className="book-info">
        <p className="book-title">{title}</p>
      </div>
    </div>
  )
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function Toast({ type, message }) {
  if (!message) return null
  return <div className={`toast toast-${type}`}>{message}</div>
}

// ── TrainingProgress ──────────────────────────────────────────────────────────
function TrainingProgress({ done, error }) {
  return (
    <div className="training-progress">
      {TRAIN_STAGES.map((s, i) => {
        // Since we don't have real-time backend progress, we show all stages as 'active' (spinning) 
        // while the API request is running, and 'done' when it finishes.
        const status = done ? 'done' : (error ? 'error' : 'active')
        return (
          <div key={i} className={`train-step train-step--${status}`}>
            <span className="train-step-icon">
              {status === 'done' ? '✅'
                : status === 'active' ? <span className="spinner-sm" />
                  : '❌'}
            </span>
            <span className="train-step-label">{s.icon} {s.label}</span>
            {status === 'active' && <span className="train-step-detail">{s.detail}</span>}
          </div>
        )
      })}
    </div>
  )
}
// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const [books, setBooks] = useState([])
  const [selectedBook, setSelectedBook] = useState('')
  const [recommendations, setRecommendations] = useState([])

  const [mode, setMode] = useState('inference')

  const [loadingBooks, setLoadingBooks] = useState(true)
  const [loadingRec, setLoadingRec] = useState(false)
  const [training, setTraining] = useState(false)
  const [trainDone, setTrainDone] = useState(false)
  const [trainError, setTrainError] = useState(false)
  const [evalResults, setEvalResults] = useState(null)
  const [zipUrl, setZipUrl] = useState(DEFAULT_ZIP_URL)

  // Model Versioning State
  const [modelVersions, setModelVersions] = useState([])
  const [selectedVersion, setSelectedVersion] = useState('')
  const [loadedVersion, setLoadedVersion] = useState(null)
  const [loadingModels, setLoadingModels] = useState(true)
  const [loadingSpecificModel, setLoadingSpecificModel] = useState(false)

  const [recStatus, setRecStatus] = useState(null)

  const fetchBooks = () => {
    setLoadingBooks(true)
    fetch(`${API}/books`)
      .then(r => r.json())
      .then(data => {
        setBooks(data.books || [])
        if (data.books?.length) setSelectedBook(data.books[0])
      })
      .catch(() => setRecStatus({ type: 'error', message: '⚠️ Could not connect to API server. Is it running on port 8000?' }))
      .finally(() => setLoadingBooks(false))
  }

  const fetchModels = (justTrained = false) => {
    setLoadingModels(true)
    fetch(`${API}/models`)
      .then(r => r.json())
      .then(data => {
        const versions = data.models || []
        setModelVersions(versions)
        if (versions.length > 0) {
          const newestVersion = versions[0].version
          setSelectedVersion(newestVersion)

          if (justTrained) {
            // If we just finished training, the model is already in the backend's RAM
            // Also sync the loadedVersion to the exact registered version number
            setLoadedVersion(newestVersion)
          } else {
            // On initial boot, just show the metrics, model is NOT loaded
            if (versions[0].metrics && Object.keys(versions[0].metrics).length > 0) {
              setEvalResults(versions[0].metrics)
              setTrainDone(true)
              setMode('training')
            }
          }
        }
      })
      .catch(e => console.error("Could not fetch models", e))
      .finally(() => setLoadingModels(false))
  }

  useEffect(() => {
    fetchBooks()
    fetchModels()
  }, [])

  // ── Recommend ──
  const handleRecommend = async () => {
    if (!selectedBook) return
    setLoadingRec(true)
    setRecommendations([])
    setRecStatus({ type: 'loading', message: '🔍 Finding similar books...' })
    try {
      const res = await fetch(`${API}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ book_name: selectedBook }),
      })
      if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Recommendation failed') }
      const data = await res.json()
      setRecommendations(data.recommendations || [])
      setRecStatus(null)
    } catch (e) {
      setRecStatus({ type: 'error', message: `❌ ${e.message}` })
    } finally {
      setLoadingRec(false)
    }
  }

  // ── Train ──
  const handleTrain = async () => {
    setTraining(true)
    setTrainDone(false)
    setTrainError(false)

    try {
      const res = await fetch(`${API}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_url: zipUrl.trim() || null }),
      })
      if (!res.ok) throw new Error('Training failed')

      const data = await res.json()
      if (data.evaluation && !data.evaluation.error) {
        setEvalResults(data.evaluation)
      } else {
        setEvalResults(null)
      }

      setTrainDone(true)

      // Immediately mark model as loaded in RAM (training puts it in memory directly)
      // Use a sentinel so the warning disappears right away before fetchModels resolves
      setLoadedVersion('__training_complete__')
      setSelectedVersion('__training_complete__')

      // Refresh books and models after training
      fetchBooks()
      fetchModels(true) // Will update to real version number once DagsHub registers it
    } catch (e) {
      setTrainError(true)
    } finally {
      setTraining(false)
    }
  }

  // ── Load specific Model Version ──
  const handleLoadModel = async () => {
    if (!selectedVersion) return
    setLoadingSpecificModel(true)
    setTrainDone(false)
    setTrainError(false)

    // Switch tracking mode to 'training' so we can show the report cleanly
    setMode('training')

    try {
      const res = await fetch(`${API}/models/load/${selectedVersion}`, { method: 'POST' })
      if (!res.ok) throw new Error('Failed to load model')
      const data = await res.json()

      if (data.evaluation && !data.evaluation.error) {
        setEvalResults(data.evaluation)
      } else {
        setEvalResults(null)
      }

      setTrainDone(true)
      setLoadedVersion(selectedVersion) // Mark this specifically as loaded
      fetchBooks() // refresh the inference books corresponding to this loaded model
    } catch (e) {
      console.error(e)
      setTrainError(true)
    } finally {
      setLoadingSpecificModel(false)
    }
  }

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-left">
          <h1>📚 Book Recommender System</h1>
          <p>Collaborative filtering · Powered by KNN · End-to-end ML pipeline</p>
        </div>
        <Link to="/logs" className="btn-logs">🗂️ Logs</Link>
      </header>

      <div style={{ display: 'flex', gap: '20px', maxWidth: '1200px', margin: '0 auto', padding: '0 20px', alignItems: 'flex-start' }}>
        {/* ── Sidebar ── */}
        <aside style={{ flex: '0 0 300px', backgroundColor: 'var(--surface-color)', padding: '20px', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
          <h2 style={{ fontSize: '1.2rem', marginTop: 0, color: 'var(--primary-color)' }}>Model Registry</h2>
          <p style={{ fontSize: '0.9rem', color: '#888', marginBottom: '20px' }}>Select a DagsHub model version to download and use for inference.</p>

          {loadingModels ? (
            <div className="toast toast-loading" style={{ margin: 0 }}>Loading versions...</div>
          ) : modelVersions.length === 0 ? (
            <div className="toast toast-error" style={{ margin: 0 }}>No models found. Train one first!</div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
              <select
                className="book-select"
                value={selectedVersion}
                onChange={e => setSelectedVersion(e.target.value)}
              >
                {modelVersions.map(m => {
                  const displayStatus = (m.status && m.status !== 'None') ? ` (${m.status})` : ''
                  return (
                    <option key={m.version} value={m.version}>
                      Version {m.version}{displayStatus}
                    </option>
                  )
                })}
              </select>
              {selectedVersion !== loadedVersion && (
                <div style={{ fontSize: '0.85rem', color: '#ff9800', lineHeight: '1.4' }}>
                  ⚠️ To use this model for <strong>Inference</strong>, you must click Load Selected Model below.
                </div>
              )}
              <button
                className="btn btn-primary"
                onClick={handleLoadModel}
                disabled={loadingSpecificModel || training}
              >
                {loadingSpecificModel ? <span className="spinner-sm" /> : '📥'}
                {loadingSpecificModel ? ' Downloading...' : ' Load Selected Model'}
              </button>
            </div>
          )}
        </aside>

        {/* ── Main Content ── */}
        <main className="container" style={{ flex: 1, margin: 0 }}>
          {/* ── Mode Toggle ── */}
          <div className="mode-toggle" style={{ display: 'flex', gap: '10px', marginBottom: '20px', justifyContent: 'center' }}>
            <button className={`btn ${mode === 'training' ? 'btn-primary' : ''}`} onClick={() => setMode('training')}>
              Training
            </button>
            <button className={`btn ${mode === 'inference' ? 'btn-primary' : ''}`} onClick={() => setMode('inference')}>
              Inference
            </button>
          </div>

          {/* ── Controls ── */}
          <section className="controls">
            {/* Train */}
            {mode === 'training' && (
              <div>
                <div style={{ marginBottom: '12px', fontSize: '1rem', fontWeight: '500', color: 'var(--primary-color)' }}>
                  {modelVersions.length === 0 ? '⚠️ Train your model first' : '⚠️ Retrain model if new data is present'}
                </div>
                <div style={{ marginBottom: '16px' }}>
                  <label htmlFor="zip-url-input" style={{ display: 'block', fontSize: '0.85rem', color: '#aaa', marginBottom: '6px' }}>
                    📦 Dataset ZIP URL
                  </label>
                  <input
                    id="zip-url-input"
                    type="text"
                    value={zipUrl}
                    onChange={e => setZipUrl(e.target.value)}
                    placeholder="https://... (zip file URL)"
                    disabled={training}
                    style={{
                      width: '100%',
                      fontSize: '0.85rem',
                      fontFamily: 'monospace',
                      padding: '12px 16px',
                      borderRadius: '10px',
                      border: '1px solid var(--border)',
                      background: 'var(--surface)',
                      color: 'var(--text)',
                      outline: 'none',
                      boxSizing: 'border-box',
                    }}
                  />
                </div>
                <button className="btn btn-train" onClick={handleTrain} disabled={training}>
                  {training ? <span className="spinner" /> : '🔧'}
                  {training ? 'Training...' : 'Train Recommender System'}
                </button>

                {/* Stage-by-stage progress (Server is running the entire pipeline) */}
                {(training || trainDone || trainError) && (
                  <div style={{ marginTop: 16 }}>
                    <p style={{ color: '#888', fontSize: '0.9rem', marginBottom: 8 }}>Executing Pipeline via API...</p>
                    <TrainingProgress done={trainDone} error={trainError} />
                  </div>
                )}

                {trainDone && <div className="toast toast-success" style={{ marginTop: 12 }}>✅ Training completed successfully!</div>}
                {trainError && <div className="toast toast-error" style={{ marginTop: 12 }}>❌ Training failed. Check Logs for details.</div>}

                {/* Evaluation Results */}
                {trainDone && evalResults && (
                  <div className="eval-results" style={{ marginTop: 24, padding: 20, backgroundColor: 'var(--surface-color)', borderRadius: 12, border: '1px solid var(--border-color)' }}>
                    <h3 style={{ marginTop: 0, marginBottom: 16, color: 'var(--primary-color)' }}>📊 Model Evaluation Report</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                      <div className="eval-metric">
                        <span style={{ fontSize: '0.9rem', color: '#888' }}>Users Tested</span>
                        <strong style={{ display: 'block', fontSize: '1.2rem' }}>{evalResults.users_tested}</strong>
                      </div>
                      <div className="eval-metric">
                        <span style={{ fontSize: '0.9rem', color: '#888' }}>Total Hits</span>
                        <strong style={{ display: 'block', fontSize: '1.2rem' }}>{evalResults.total_hits}</strong>
                      </div>
                      <div className="eval-metric">
                        <span style={{ fontSize: '0.9rem', color: '#888' }}>Hit Ratio @ {evalResults.recommendations_count}</span>
                        <strong style={{ display: 'block', fontSize: '1.2rem', color: 'var(--success-color)' }}>{(evalResults.hit_ratio * 100).toFixed(2)}%</strong>
                      </div>
                      <div className="eval-metric">
                        <span style={{ fontSize: '0.9rem', color: '#888' }}>Automated Precision</span>
                        <strong style={{ display: 'block', fontSize: '1.2rem' }}>{(evalResults.precision * 100).toFixed(2)}%</strong>
                      </div>
                      <div className="eval-metric">
                        <span style={{ fontSize: '0.9rem', color: '#888' }}>Automated Recall</span>
                        <strong style={{ display: 'block', fontSize: '1.2rem' }}>{(evalResults.recall * 100).toFixed(2)}%</strong>
                      </div>
                      <div className="eval-metric">
                        <span style={{ fontSize: '0.9rem', color: '#888' }}>Ranking Quality (NDCG)</span>
                        <strong style={{ display: 'block', fontSize: '1.2rem' }}>{(evalResults.ndcg * 100).toFixed(2)}%</strong>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Book selector */}
            {mode === 'inference' && (
              <>
                <p className="controls-title">🎯 Find Similar Books</p>
                <div className="control-row">
                  <div className="select-wrapper">
                    <label htmlFor="book-select">Type or select a book</label>
                    {loadingBooks ? (
                      <div className="toast toast-loading">Loading book list...</div>
                    ) : books.length === 0 ? (
                      <div className="no-books-msg">
                        ⚠️ No model loaded — please <strong>Load a Model from the Registry</strong> or <strong>Train a new Model</strong>.
                      </div>
                    ) : (
                      <select
                        id="book-select"
                        className="book-select"
                        value={selectedBook}
                        onChange={e => { setSelectedBook(e.target.value); setRecommendations([]); setRecStatus(null) }}
                      >
                        {books.map(b => <option key={b} value={b}>{b}</option>)}
                      </select>
                    )}
                  </div>
                  <button
                    className="btn btn-primary"
                    onClick={handleRecommend}
                    disabled={loadingRec || !selectedBook || loadingBooks || books.length === 0}
                  >
                    {loadingRec ? <span className="spinner" /> : '✨'}
                    {loadingRec ? 'Searching...' : 'Show Recommendations'}
                  </button>
                </div>
                <Toast type={recStatus?.type} message={recStatus?.message} />
              </>
            )}
          </section>

          {/* ── Results ── */}
          {mode === 'inference' && recommendations.length > 0 && (
            <section>
              <h2 className="recommendations-title">✨ Recommended for "{selectedBook}"</h2>
              <div className="books-grid">
                {recommendations.map((book, i) => (
                  <BookCard key={i} title={book.title} posterUrl={book.poster_url} delay={i * 0.07} />
                ))}
              </div>
            </section>
          )}

          {mode === 'inference' && !loadingRec && recommendations.length === 0 && !recStatus && books.length > 0 && (
            <div className="empty-state">
              <div className="emoji">📖</div>
              <p>Select a book and click <strong>Show Recommendations</strong> to discover similar reads.</p>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
