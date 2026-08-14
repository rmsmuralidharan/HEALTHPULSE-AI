import os, sys
import pandas as pd

from HealthPulse_AI_project.exception.exception import HealthPulseException
from HealthPulse_AI_project.logging.logger import logging

class DataIngestion:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def initiate_data_ingestion(self):
        try:
            logging.info('Starting PTB-XL data ingestion')

            database_path = os.path.join(
                self.data_path,
                "ptbxl_database.csv"
            )

            scp_path = os.path.join(
                self.data_path,
                "scp_statements.csv"
            )

            ## check whether required files exist
            if not os.path.exists(database_path):
                raise FileNotFoundError(
                    f"PTB-XL database file not found: {database_path}"
                )

            if not os.path.exists(scp_path):
                raise FileNotFoundError(
                    f"SCP statements file not found: {scp_path}"
                )

            ## load PTB-XL metadata
            database_df = pd.read_csv(database_path)

            ## load diagnostic statement
            scp_df = pd.read_csv(scp_path)

            logging.info(
                f"PTB-XL database shape: {database_df.shape}, "
                f"SCP statements shape: {scp_df.shape}"
            )

            logging.info(
                 "PTB-XL data ingestion completed successfully"
            )

            return database_df, scp_df
        except Exception as e:
            logging.exception(
                "Error Ocurred during PTB-XL data ingestion"
            )

            raise HealthPulseException(e, sys)
