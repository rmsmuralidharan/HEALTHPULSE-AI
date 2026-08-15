import os

from HealthPulse_AI_project.components.data_ingestion import DataIngestion
from HealthPulse_AI_project.components.target_creation import TargetCreation
from HealthPulse_AI_project.components.data_splitting import DataSplitting



if __name__ == "__main__":

    # ==========================================
    # 1. PTB-XL dataset path
    # ==========================================

    data_path = os.path.join(
        "data",
        "raw",
        "ptbxl"
    )

    # ==========================================
    # 2. Data ingestion
    # ==========================================

    ingestion = DataIngestion(
        data_path=data_path
    )

    database_df, scp_df = (
        ingestion.initiate_data_ingestion()
    )

    # ==========================================
    # 3. Target creation
    # ==========================================

    target_creator = TargetCreation()

    target_df = (
        target_creator.initiate_target_creation(
            database_df,
            scp_df
        )
    )

        # ==========================================
    # 4. Patient-level splitting
    # ==========================================

    splitter = DataSplitting(
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
        random_state=42
    )

    (
        train_df,
        validation_df,
        test_df
    ) = splitter.initiate_data_splitting(
        target_df
    )

    # ==========================================
    # Helper function
    # ==========================================

    def print_split_details(
        name,
        split_df
    ):

        records = len(split_df)

        patients = (
            split_df["patient_id"]
            .nunique()
        )

        mi_count = (
            split_df["Target"] == 1
        ).sum()

        non_mi_count = (
            split_df["Target"] == 0
        ).sum()

        mi_percentage = (
            mi_count / records
        ) * 100

        print(f"\n{name}")
        print("-" * 60)

        print(
            f"Records : {records}"
        )

        print(
            f"Patients: {patients}"
        )

        print(
            f"MI      : {mi_count}"
        )

        print(
            f"Non-MI  : {non_mi_count}"
        )

        print(
            f"MI %    : {mi_percentage:.2f}%"
        )

    # ==========================================
    # Display split results
    # ==========================================

    print("\n")
    print("=" * 60)
    print("PATIENT-LEVEL DATA SPLIT RESULTS")
    print("=" * 60)

    print_split_details(
        "TRAIN",
        train_df
    )

    print_split_details(
        "VALIDATION",
        validation_df
    )

    print_split_details(
        "TEST",
        test_df
    )

    # ==========================================
    # Patient overlap validation
    # ==========================================

    train_patients = set(
        train_df["patient_id"]
    )

    validation_patients = set(
        validation_df["patient_id"]
    )

    test_patients = set(
        test_df["patient_id"]
    )

    train_validation_overlap = (
        train_patients
        &
        validation_patients
    )

    train_test_overlap = (
        train_patients
        &
        test_patients
    )

    validation_test_overlap = (
        validation_patients
        &
        test_patients
    )

    print("\n")
    print("=" * 60)
    print("PATIENT OVERLAP VALIDATION")
    print("=" * 60)

    print(
        f"Train ∩ Validation : "
        f"{len(train_validation_overlap)}"
    )

    print(
        f"Train ∩ Test       : "
        f"{len(train_test_overlap)}"
    )

    print(
        f"Validation ∩ Test  : "
        f"{len(validation_test_overlap)}"
    )

    # ==========================================
    # Final validation
    # ==========================================

    split_validation = (
        len(train_validation_overlap) == 0
        and
        len(train_test_overlap) == 0
        and
        len(validation_test_overlap) == 0
    )

    print(
        f"\nSplit validation: "
        f"{'PASS' if split_validation else 'FAIL'}"
    )

    print("=" * 60)
