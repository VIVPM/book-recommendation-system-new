import os
import sys
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make the books_recommender package importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from books_recommender.config.configuration import AppConfiguration
from books_recommender.pipeline.training_pipeline import TrainingPipeline
from books_recommender.logger.log import logging
from books_recommender.exception.exception_handler import AppException
import evaluate

# -------------------------------------------------------------------
app = FastAPI(title="Book Recommender API", version="1.0.0")

# Allow requests from the React dev server (and any origin in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Startup Event
# -------------------------------------------------------------------

def _load_artifacts():
    """Load recommendation artifacts from disk."""
    config = AppConfiguration().get_recommendation_config()
    try:
        book_pivot  = pickle.load(open(config.book_pivot_serialized_objects, "rb"))
        final_rating = pickle.load(open(config.final_rating_serialized_objects, "rb"))
        model        = pickle.load(open(config.trained_model_path, "rb"))
        book_names   = pickle.load(open(config.book_name_serialized_objects, "rb"))
        return book_pivot, final_rating, model, book_names
    except FileNotFoundError:
        logging.warning("Artifacts not found on startup. Model needs to be trained.")
        return None, None, None, None

@app.on_event("startup")
async def startup():
    """Load artifacts into memory once when server starts."""
    logging.info("Starting up FastAPI server...")
    logging.info("Loading ML artifacts into memory...")
    artifacts = _load_artifacts()
    app.state.artifacts = artifacts
    if artifacts[0] is not None:
        logging.info("Artifacts loaded successfully!")
    else:
        logging.warning("No artifacts loaded. Training required.")

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _fetch_poster(suggestion, book_pivot, final_rating):
    """Return poster image URLs for the suggested book indices."""
    poster_url = []
    for book_id in suggestion:
        book_name = book_pivot.index[book_id]
        idx = np.where(final_rating["title"] == book_name)[0][0]
        poster_url.append(final_rating.iloc[idx]["image_url"])
    return poster_url


# -------------------------------------------------------------------
# Request / Response models
# -------------------------------------------------------------------

class RecommendRequest(BaseModel):
    book_name: str


class RecommendedBook(BaseModel):
    title: str
    poster_url: str


class RecommendResponse(BaseModel):
    recommendations: list[RecommendedBook]


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.get("/health", summary="Health Check")
def health():
    """Health check for Docker/production monitoring."""
    model_loaded = hasattr(app.state, 'artifacts') and app.state.artifacts[2] is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded
    }

@app.get("/books", summary="Get all book names for the dropdown")
def get_books():
    """Returns list of all book names available for recommendation."""
    try:
        if not hasattr(app.state, 'artifacts') or app.state.artifacts[3] is None:
            return {"books": []}
            
        book_names = app.state.artifacts[3]
        return {"books": list(book_names)}
    except Exception as e:
        logging.error(f"Error loading book names: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend", response_model=RecommendResponse, summary="Get 5 book recommendations")
def recommend(request: RecommendRequest):
    """
    Given a book name, returns 5 similar book recommendations with poster images.
    """
    try:
        book_pivot, final_rating, model, book_names = app.state.artifacts
        
        if book_pivot is None:
             raise HTTPException(status_code=400, detail="Model is not trained yet. Please hit /train first.")

        if request.book_name not in book_pivot.index:
            raise HTTPException(status_code=404, detail=f"Book '{request.book_name}' not found.")

        book_id = np.where(book_pivot.index == request.book_name)[0][0]
        distances, suggestion = model.kneighbors(
            book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6
        )

        results = []
        for i in range(1, 6):          # skip index 0 — it's the input book itself
            title = book_pivot.index[suggestion[0][i]]
            # Get poster
            idx_arr = np.where(final_rating["title"] == title)[0]
            poster = final_rating.iloc[idx_arr[0]]["image_url"] if len(idx_arr) > 0 else ""
            results.append(RecommendedBook(title=title, poster_url=poster))

        logging.info(f"Recommendations served for: {request.book_name}")
        return RecommendResponse(recommendations=results)

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", summary="Trigger the full training pipeline")
def train():
    """
    Runs data ingestion → validation → transformation → model training.
    This may take several minutes.
    """
    try:
        logging.info("Training pipeline triggered via API.")
        pipeline = TrainingPipeline()
        pipeline.start_training_pipeline()
        
        # Reload artifacts into memory after training
        app.state.artifacts = _load_artifacts()
        
        # Run evaluation
        logging.info("Running evaluation after training...")
        try:
            book_pivot, model, final_rating = evaluate.load_ml_components()
            eval_results = evaluate.test_recommendation_system(book_pivot, model, final_rating, number_of_recommendations=5)
        except Exception as eval_e:
            logging.error(f"Evaluation failed: {eval_e}")
            eval_results = {"error": str(eval_e)}
        
        logging.info("Training pipeline completed and new artifacts loaded via API.")
        return {
            "message": "Training completed successfully.",
            "evaluation": eval_results
        }
    except Exception as e:
        logging.error(f"Training pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs", summary="Get log files from backend/logs/")
def get_logs():
    """
    Returns all log files from the backend/logs/ directory sorted newest-first.
    Each entry has: filename, content (list of lines).
    """
    try:
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        if not os.path.isdir(logs_dir):
            return {"logs": []}

        files = sorted(
            [f for f in os.listdir(logs_dir) if f.endswith(".log")],
            reverse=True   # newest first (filenames contain timestamp)
        )

        result = []
        for fname in files:
            fpath = os.path.join(logs_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.read().splitlines()
                result.append({"filename": fname, "lines": lines})
            except Exception:
                result.append({"filename": fname, "lines": ["[Could not read file]"]})

        return {"logs": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
