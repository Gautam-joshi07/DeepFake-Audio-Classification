import os
import zipfile
import gdown
import shutil
from src.Classifier import logger
from src.Classifier.utils.common import get_size
from src.Classifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
     
    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try:
            source_path = self.config.local_source_path
            destination_path = self.config.local_data_file
            os.makedirs(self.config.root_dir, exist_ok=True)
            logger.info(f"Copying data from {source_path} to {destination_path}")
            shutil.copy(source_path, destination_path)
            logger.info(f"Copied data from {source_path} to {destination_path}")
        except Exception as e:
            raise e

        
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)