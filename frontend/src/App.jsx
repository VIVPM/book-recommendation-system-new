import { useState, useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import './App.css'

const API = 'http://localhost:8000'

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
function TrainingProgress({ stageIndex, done, error }) {
  return (
    <div className="training-progress">
      {TRAIN_STAGES.map((s, i) => {
        const status = done ? 'done'
          : error ? (i <= stageIndex ? 'error' : 'idle')
            : i < stageIndex ? 'done'
              : i === stageIndex ? 'active'
                : 'idle'
        return (
          <div key={i} className={`train-step train-step--${status}`}>
            <span className="train-step-icon">
              {status === 'done' ? '✅'
                : status === 'active' ? <span className="spinner-sm" />
                  : status === 'error' ? '❌'
                    : '⏳'}
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

  const [loadingBooks, setLoadingBooks] = useState(true)
  const [loadingRec, setLoadingRec] = useState(false)
  const [training, setTraining] = useState(false)
  const [trainStage, setTrainStage] = useState(0)
  const [trainDone, setTrainDone] = useState(false)
  const [trainError, setTrainError] = useState(false)

  const [recStatus, setRecStatus] = useState(null)
  const stageTimerRef = useRef(null)

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

  useEffect(() => { fetchBooks() }, [])

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
    setTrainStage(0)

    // Simulate stage advancement while API call runs
    // Each stage gets ~20s before advancing (real timing may differ)
    const stageDurations = [20000, 30000, 25000, 15000]
    let currentStage = 0
    const advance = () => {
      currentStage += 1
      if (currentStage < TRAIN_STAGES.length) {
        setTrainStage(currentStage)
        stageTimerRef.current = setTimeout(advance, stageDurations[currentStage])
      }
    }
    stageTimerRef.current = setTimeout(advance, stageDurations[0])

    try {
      const res = await fetch(`${API}/train`, { method: 'POST' })
      clearTimeout(stageTimerRef.current)
      if (!res.ok) throw new Error('Training failed')
      setTrainStage(TRAIN_STAGES.length - 1)
      setTrainDone(true)
      // Refresh books after training
      fetchBooks()
    } catch (e) {
      clearTimeout(stageTimerRef.current)
      setTrainError(true)
    } finally {
      setTraining(false)
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

      <main className="container">
        {/* ── Controls ── */}
        <section className="controls">
          {/* Train */}
          <div>
            <button className="btn btn-train" onClick={handleTrain} disabled={training}>
              {training ? <span className="spinner" /> : '🔧'}
              {training ? 'Training...' : 'Train Recommender System'}
            </button>

            {/* Stage-by-stage progress */}
            {(training || trainDone || trainError) && (
              <TrainingProgress
                stageIndex={trainStage}
                done={trainDone}
                error={trainError}
              />
            )}

            {trainDone && <div className="toast toast-success" style={{ marginTop: 12 }}>✅ Training completed successfully!</div>}
            {trainError && <div className="toast toast-error" style={{ marginTop: 12 }}>❌ Training failed. Check Logs for details.</div>}
          </div>

          {/* Book selector */}
          <p className="controls-title">🎯 Find Similar Books</p>
          <div className="control-row">
            <div className="select-wrapper">
              <label htmlFor="book-select">Type or select a book</label>
              {loadingBooks ? (
                <div className="toast toast-loading">Loading book list...</div>
              ) : books.length === 0 ? (
                <div className="no-books-msg">
                  ⚠️ No books found — please <strong>Train the Recommender System</strong> first.
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
        </section>

        {/* ── Results ── */}
        {recommendations.length > 0 && (
          <section>
            <h2 className="recommendations-title">✨ Recommended for "{selectedBook}"</h2>
            <div className="books-grid">
              {recommendations.map((book, i) => (
                <BookCard key={i} title={book.title} posterUrl={book.poster_url} delay={i * 0.07} />
              ))}
            </div>
          </section>
        )}

        {!loadingRec && recommendations.length === 0 && !recStatus && books.length > 0 && (
          <div className="empty-state">
            <div className="emoji">📖</div>
            <p>Select a book and click <strong>Show Recommendations</strong> to discover similar reads.</p>
          </div>
        )}
      </main>
    </div>
  )
}
