import os

from HealthPulse_AI_project.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    data_path = os.path.join(
        "data",
        "raw",
        "ptbxl"
    )

    ingestion = DataIngestion(
        data_path=data_path
    )

    database_df, scp_df = (
        ingestion.initiate_data_ingestion()
    )

    print("\nPTB-XL Database Shape:")
    print(database_df.shape)

    print("\nPTB-XL Database Columns:")
    print(database_df.columns.tolist())

    print("\nFirst 5 Database Records:")
    print(database_df.head())

    print("\nSCP Statements Shape:")
    print(scp_df.shape)

    print("\nFirst 5 SCP Statements:")
    print(scp_df.head())