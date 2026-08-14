import os, sys, ast
from collections import Counter
import pandas as pd

from HealthPulse_AI_project.exception.exception import HealthPulseException
from HealthPulse_AI_project.logging.logger import logging

class DataValidation:
    def __init__(self, data_path:str):
        self.data_path = data_path

    def initiate_metadata(self, df: pd.DataFrame):
        try:
            logging.info("Starting PTB-XL data validation")

            validation_results = {}

            ## 1. dataset Shape

            validation_results['rows'] = len(df)
            validation_results['columns'] = len(df.columns)

            logging.info(
                f"Dataset shape: {df.shape}"
            )


            ## 2. Required columns

            required_columns = [
                "ecg_id",
                "patient_id",
                "age",
                "sex",
                "height",
                "weight",
                "scp_codes",
                "filename_lr",
                "filename_hr"               
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in df.columns
            ]

            validation_results['missing_columns'] = missing_columns

            logging.info(
                f"Missing required columns: {missing_columns}"
            )

            # stop safely if required columns are missing
            if missing_columns:
                logging.warning(
                    "Required columns are missing. "
                    "Validation cannot continue safely."
                )


            ## Duplicate ECG IDs

            duplicate_ecg_ids = (
                df['ecg_id'].duplicated().sum()
            )                         

            validation_results['duplicate_ecg_ids'] = int(duplicate_ecg_ids)

            logging.info(
                f"Duplicate ECG IDs: {duplicate_ecg_ids}"
            )

            ## 4. Patient-level information

            unique_patients = (
                df['patient_id'].nunique()
            )          

            patients_with_multiple_ecgs = (
                df['patient_id']
                .value_counts()
                .gt(1)
                .sum()
            )

            max_ecgs_per_patient = (
                df["patient_id"]
                .value_counts()
                .max()
            )

            validation_results['unique_patients'] = int(unique_patients) 

            validation_results['patients_with_multiple_ecgs'] = int(patients_with_multiple_ecgs)

            validation_results['max_ecgs_per_patient'] = int(max_ecgs_per_patient)

            logging.info(
                f"Unique patients: {unique_patients}"
            )

            logging.info(
                "Patients with multiple ECGs: "
                f"{patients_with_multiple_ecgs}"
            )

            logging.info(
                f"Maximum ECGs per patient: "
                f"{max_ecgs_per_patient}"
            )    

            # 5. Missing values

            missing_values = df.isnull().sum()

            validation_results['missing_values'] = missing_values.to_dict()

            logging.info(
                "Missing-value validation completed"
            )     

            # 6. Missing patient IDs             
             
            missing_patient_id = (
                df['patient_id'].isna().sum()
            )

            validation_results['missing_patient_id'] = int(missing_patient_id)

            logging.info(
                f"Missing patient IDs: {missing_patient_id}"
            )

            # 7. Age validation

            invalid_age = (
                (df['age'] < 0) | (df['age'] > 100)
            ).sum()

            validation_results['invalid_age'] = int(invalid_age)

            logging.info(
                f"Age outside expected range: {invalid_age}"
            )

            # 8. Sex validation

            valid_sex_values = [0,1]

            invalid_sex_value = (
                ~df['sex'].isin(valid_sex_values) 
                & df['sex'].notna()
            ).sum()

            validation_results['invalid_sex_values'] = int(invalid_sex_value)

            logging.info(
                f"Invalid sex values: {invalid_sex_value}"
            )

            # 9. SCP code validation

            scp_counter = Counter()
            invalid_scp_rows = 0

            for codes in df['scp_codes']:
                if pd.isna(codes):
                    continue

                try:
                    # Convert string representation
                    # into Python dictionary

                    parsed_codes = ast.literal_eval(codes)

                    if isinstance(parsed_codes, dict):

                        scp_counter.update(
                            parsed_codes.keys()
                        )                  
                    else:
                        invalid_scp_rows += 1
 
                except (ValueError, SyntaxError, TypeError):
                    invalid_scp_rows += 1

            validation_results['invalid_scp_rows'] = int(invalid_scp_rows)

            validation_results['unique_scp_codes'] = len(scp_counter)

            validation_results['scp_code_frequency'] = dict(scp_counter.most_common())

            logging.info(
                f"Unique SCP codes: {len(scp_counter)}"
            )

            logging.info(
                f"Invalid SCP records: {invalid_scp_rows}"
            ) 

            # 10. LR ECG file validation

            missing_lr_files = 0

            for filename in df['filename_lr'].dropna():
                hea_path = os.path.join(
                    self.data_path,
                    filename + '.hea'
                )                    

                dat_path = os.path.join(
                    self.data_path,
                    filename + '.dat'
                )                    

                if (
                    not os.path.exists(hea_path)
                    or 
                    not os.path.exists(dat_path)
                ):
                    missing_lr_files += 1

            validation_results['missing_lr_files'] = int(missing_lr_files)   

            logging.info(
                f"Missing LR ECG files: {missing_lr_files}"
            )

            # 11. HR ECG file validation

            missing_hr_files = 0

            for filename in df['filename_hr'].dropna():
                hea_path = os.path.join(
                    self.data_path,
                    filename + '.hea'
                )           

                dat_path = os.path.join(
                    self.data_path,
                    filename + '.dat'
                )

                if (
                    not os.path.exists(hea_path)
                    or
                    not os.path.exists(dat_path)
                ):
                    missing_hr_files += 1

            validation_results['missing_hr_files'] = int(missing_hr_files)

            logging.info(
                f"Missing HR ECG files: {missing_hr_files}"
            )

            # 12. Final validation status

            validation_passed = (
                len(missing_columns) == 0
                and duplicate_ecg_ids == 0
                and missing_patient_id == 0
                and invalid_sex_value == 0
                and invalid_scp_rows ==0
                and missing_lr_files == 0
                and missing_hr_files == 0
            )        

            if validation_passed:
                validation_results['validation_status'] = 'PASS'

            else:
                validation_results['validation_status'] = 'CHECK_REQUIRED'

            logging.info(
                "PTB-XL data validation completed"
            )

            logging.info(
                "Validation status: "
                f"{validation_results['validation_status']}"
            )

            return validation_results
                        
        except Exception as e:

            logging.exception(
                "Error ocurred during PTB-XL data validation"
            )

            raise HealthPulseException(e, sys)