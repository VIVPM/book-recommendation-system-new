import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from books_recommender.pipeline.training_pipeline import TrainingPipeline


if __name__ == "__main__":
    training_pipeline = TrainingPipeline()
    training_pipeline.start_training_pipeline()