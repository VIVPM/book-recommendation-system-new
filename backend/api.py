import os
import sys

# Fix Windows charmap encoding error with dagshub/mlflow emoji output
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

import pickle
import numpy as np
import shutil
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
import mlflow
from mlflow.tracking import MlflowClient
import dagshub

# Initialize DagsHub MLflow tracking globally
try:
    dagshub.init(repo_owner='vivpm99', repo_name='book-recommendation-system-new', mlflow=True)
except Exception as e:
    logging.warning(f"Could not initialize DagsHub globally: {e}")

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
    """Start server with empty artifacts. DagsHub models must be explicitly loaded."""
    logging.info("Starting up FastAPI server...")
    logging.info("Waiting for explicit model load from DagsHub via /models/load...")
    app.state.artifacts = (None, None, None, None)
    logging.warning("No artifacts loaded locally on boot to enforce DagsHub tracking.")

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
             raise HTTPException(status_code=400, detail="Model is not loaded. Please select a model version from the sidebar or train a new one.")

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
        
        with mlflow.start_run(run_name="Training_Pipeline"):
            # Set a tag or basic parameter
            mlflow.set_tag("version", "1.0.0")
            
            pipeline = TrainingPipeline()
            pipeline.start_training_pipeline()
            
            # Reload artifacts into memory after training
            app.state.artifacts = _load_artifacts()
            
            # Run evaluation
            logging.info("Running evaluation after training...")
            try:
                book_pivot, model, final_rating = evaluate.load_ml_components()
                eval_results = evaluate.test_recommendation_system(book_pivot, model, final_rating, number_of_recommendations=5)
                
                # Log metrics to MLflow
                if "error" not in eval_results:
                    mlflow.log_metric("users_tested", eval_results["users_tested"])
                    mlflow.log_metric("total_hits", eval_results["total_hits"])
                    mlflow.log_metric("hit_ratio", eval_results["hit_ratio"])
                    mlflow.log_metric("precision", eval_results["precision"])
                    mlflow.log_metric("recall", eval_results["recall"])
                    mlflow.log_metric("ndcg", eval_results["ndcg"])
                    
            except Exception as eval_e:
                logging.error(f"Evaluation failed: {eval_e}")
                eval_results = {"error": str(eval_e)}
            
            # Upload the entire artifacts folder to the Experiment Run in DagsHub
            artifacts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
            logging.info(f"Uploading artifacts folder to DagsHub Run: {artifacts_dir}")
            mlflow.log_artifacts(artifacts_dir, artifact_path="model_artifacts")
            
            # Explicitly register just the KNN model to the DagsHub Model Registry
            logging.info("Registering model.pkl to Model Registry...")
            # We already loaded 'model' in the try block above, but if it failed, we shouldn't register
            if "error" not in eval_results and model is not None:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="knn_model",
                    registered_model_name="Book_Recommender_Model"
                )
            
        logging.info("Training pipeline completed and new artifacts loaded via API.")
        return {
            "message": "Training completed successfully and tracked in DagsHub.",
            "evaluation": eval_results
        }
    except Exception as e:
        logging.error(f"Training pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", summary="Get all registered model versions")
def get_models():
    """Returns list of all available model versions from DagsHub Model Registry."""
    try:
        client = MlflowClient()
        # Search for all versions of our registered model
        versions = client.search_model_versions("name='Book_Recommender_Model'")
        
        result = []
        for v in versions:
            # Get the run to extract metrics
            try:
                run = client.get_run(v.run_id)
                metrics = run.data.metrics
            except Exception:
                metrics = {}
                
            result.append({
                "version": v.version,
                "run_id": v.run_id,
                "status": v.current_stage,
                "metrics": metrics
            })
            
        # Sort so newest version is first
        result = sorted(result, key=lambda x: int(x["version"]), reverse=True)
        return {"models": result}
    except Exception as e:
        logging.error(f"Error fetching models from registry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/load/{version}", summary="Load a specific model version from DagsHub")
def load_model(version: str):
    """Downloads the full artifact directory for a specific model version and loads it into memory."""
    try:
        client = MlflowClient()
        versions = client.search_model_versions("name='Book_Recommender_Model'")
        target_v = next((v for v in versions if v.version == version), None)
        
        if not target_v:
             raise HTTPException(status_code=404, detail=f"Model version {version} not found")
             
        run_id = target_v.run_id
        logging.info(f"Downloading artifacts for version {version} (run_id: {run_id}) from DagsHub...")
        
        # Download the 'model_artifacts' folder
        # This downloads it to a temporary local path
        download_path = client.download_artifacts(run_id, "model_artifacts")
        logging.info(f"Artifacts downloaded successfully to {download_path}")
        
        # Copy to our local backend/artifacts directory
        local_artifacts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
        if os.path.exists(local_artifacts_dir):
            shutil.rmtree(local_artifacts_dir)
            
        # The download_path points to the downloaded folder, inside it are the contents
        shutil.copytree(download_path, local_artifacts_dir)
        
        # Reload artifacts into memory
        app.state.artifacts = _load_artifacts()
        
        # Get metrics to return to the frontend
        try:
            run = client.get_run(run_id)
            eval_results = run.data.metrics
        except Exception:
            eval_results = {}
            
        logging.info(f"Version {version} loaded successfully into memory.")
        return {
            "message": f"Version {version} loaded successfully",
            "evaluation": eval_results
        }
    except Exception as e:
        logging.error(f"Error loading model version {version}: {e}")
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
