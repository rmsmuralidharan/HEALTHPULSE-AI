import os

from HealthPulse_AI_project.components.data_ingestion import DataIngestion
from HealthPulse_AI_project.components.target_creation import TargetCreation

if __name__ == "__main__":

    # 1. PTB-XL dataset path

    data_path = os.path.join(
        'data', 'raw', 'ptbxl'
    )

    # 2. Data ingestion

    ingestion = DataIngestion(data_path=data_path)

    database_df, scp_df = (
        ingestion.initiate_data_ingestion()
    )

    # 3. Target creation

    target_creator = TargetCreation()

    target_df = (
        target_creator.initiate_target_creation(
            database_df=database_df,
            scp_df=scp_df
        )
    )

    # 4. Display results

    print("\n")
    print("=" * 60)
    print("MI TARGET CREATION RESULTS")
    print("=" * 60)

    print(
        f"Total records  : "
        f"{len(target_df)}"
    )

    print(
        f"MI records     : "
        f"{(target_df['Target'] == 1).sum()}"
    )

    print(
        f"Non-MI records : "
        f"{(target_df['Target'] == 0).sum()}"
    )   


    # 5. Target distribution

    print("\nTarget distribution:")

    print(
        target_df["Target"]
        .value_counts()
        .sort_index()
    )

    # ==========================================
    # 6. Target percentages
    # ==========================================

    print("\nTarget percentages:")

    target_percentages = (
        target_df["Target"]
        .value_counts(
            normalize=True
        )
        .sort_index()
        .mul(100)
        .round(2)
    )

    print(target_percentages)

    # ==========================================
    # 7. Target values validation
    # ==========================================

    print("\nUnique target values:")

    print(
        target_df["Target"].unique()
    )

    # ==========================================
    # 8. Verify target contains only 0 and 1
    # ==========================================

    valid_target = target_df[
        "Target"
    ].isin([0, 1]).all()

    print(
        f"\nTarget validation: "
        f"{'PASS' if valid_target else 'FAIL'}"
    )

    # ==========================================
    # 9. Sample records
    # ==========================================

    print("\nSample records:")

    print(
        target_df[
            [
                "ecg_id",
                "patient_id",
                "scp_codes",
                "Target"
            ]
        ].head(10)
    )

    print("=" * 60)