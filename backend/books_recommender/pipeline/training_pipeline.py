from books_recommender.components.stage_00_data_ingestion import DataIngestion
from books_recommender.components.stage_01_data_validation import DataValidation
from books_recommender.components.stage_02_data_transformation import DataTransformation
from books_recommender.components.stage_03_model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_validation = DataValidation()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def start_training_pipeline(self, dataset_url: str = None):
        """
        Starts the training pipeline.
        :param dataset_url: Optional ZIP URL to override config.yaml default.
        :return: none
        """
        self.data_ingestion.initiate_data_ingestion(dataset_url=dataset_url)
        self.data_validation.initiate_data_validation()
        self.data_transformation.initiate_data_transformation()
        self.model_trainer.initiate_model_trainer()
