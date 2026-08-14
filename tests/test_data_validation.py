import os

from HealthPulse_AI_project.components.data_ingestion import DataIngestion
from HealthPulse_AI_project.components.data_validation import DataValidation

if __name__ == "__main__":

    # PTB-XL dataset path

    data_path = os.path.join(
        'data',
        'raw',
        'ptbxl'
    )

    # Data ingestion

    ingestion = DataIngestion(
        data_path=data_path
    )

    database_df, scp_df = (
        ingestion.initiate_data_ingestion()
    )


    # Data validation

    validator = DataValidation(
        data_path=data_path
    )

    results = validator.initiate_metadata(
        database_df
    )

    # Display validation results

    print("\n")
    print("=" * 60)
    print("PTB-XL DATA VALIDATION RESULTS")
    print("=" * 60)

    print(
        f"Dataset shape       : "
        f"({results['rows']}, {results['columns']})"
    )

    print(
        f"Unique patients     : "
        f"{results['unique_patients']}"
    )

    print(
        f"Patients with >1 ECG: "
        f"{results['patients_with_multiple_ecgs']}"
    )

    print(
        f"Max ECGs per patient: "
        f"{results['max_ecgs_per_patient']}"
    )

    print(
        f"Duplicate ECG IDs   : "
        f"{results['duplicate_ecg_ids']}"
    )

    print(
        f"Missing patient IDs : "
        f"{results['missing_patient_id']}"
    )

    print(
        f"Invalid age         : "
        f"{results['invalid_age']}"
    )

    print(
        f"Invalid sex         : "
        f"{results['invalid_sex_values']}"
    )

    print(
        f"Invalid SCP records : "
        f"{results['invalid_scp_rows']}"
    )

    print(
        f"Unique SCP codes    : "
        f"{results['unique_scp_codes']}"
    )

    print(
        f"Missing LR files    : "
        f"{results['missing_lr_files']}"
    )

    print(
        f"Missing HR files    : "
        f"{results['missing_hr_files']}"
    )

    print("=" * 60)

    print(
        f"VALIDATION STATUS   : "
        f"{results['validation_status']}"
    )

    print("=" * 60)


    # Top SCP codes

    print("\nTop 20 SCP Codes:")

    scp_frequency = results[
        "scp_code_frequency"
    ]

    for code, count in list(
        scp_frequency.items()
    )[:20]:

        print(
            f"{code:<10} : {count}"
        )