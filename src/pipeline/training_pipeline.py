from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

data_ingestion = DataIngestion()

train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()


data_transformation = DataTransformation()

train_arr, test_arr, preprocessor_obj = data_transformation.initiate_data_transformation(train_data_path,test_data_path)


model_trainer = ModelTrainer()

model_trainer.initiate_model_trainer(train_arr,test_arr)