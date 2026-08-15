import sys, ast

import pandas as pd

from HealthPulse_AI_project.exception.exception import HealthPulseException
from HealthPulse_AI_project.logging.logger import logging

class TargetCreation:
    def __init__(self):
        pass

    def initiate_target_creation(
            self, 
            database_df: pd.DataFrame,
            scp_df: pd.DataFrame
    ):
        try:
            logging.info(
                "Starting MI target creation"
            )

            # 1. Validate input dataframe

            if database_df.empty:
                raise ValueError(
                    'Database dataframe is empty'
                )

            if scp_df.empty:
                raise ValueError(
                    'SCP statements dataframe is empty'
                )

            required_columns = ['scp_codes']

            missing_columns = [
                column
                for column in required_columns
                if column not in database_df.columns
            ]

            if missing_columns:
                raise ValueError(
                    'Missing required columns: '
                    f"{missing_columns}"
                )

            if "diagnostic_class" not in scp_df.columns:
                raise ValueError(
                    'diagnostic class column is missing '
                    "from scp statements"
                )

            # 2. Identify MI diagnostic codes

            if "diagnostic_class" not in scp_df.columns:
                raise ValueError(
                    "diagnostic_class column is missing "
                    "from SCP statements"
                )

            scp_code_column = scp_df.columns[0]

            if scp_code_column is None:
                raise ValueError(
                    "unable to identify SCP code column"
                )

            mi_codes = set(
                scp_df.loc[
                    scp_df['diagnostic_class'] == "MI",
                    scp_code_column
                ].dropna().astype(str)
            )

            if not mi_codes:
                raise ValueError(
                    "No MI diagnostic codes found "
                    "in SCP statements"
                )

            logging.info(
                f"MI diagnostic codes identified: "
                f"{len(mi_codes)}"
            )

            logging.info(
                f"MI codes: {sorted(mi_codes)}"
            )            

            # 3. Parse SCP codes

            def extract_scp_codes(codes):
                if pd.isna(codes):
                    return set()

                if isinstance(codes, dict):
                    return set(codes.keys())

                try:
                    parced_codes = ast.literal_eval(
                        codes
                    )

                    if isinstance(
                        parced_codes,
                        dict
                    ):
                        return set(parced_codes.keys())
                    
                except (
                   ValueError,
                   SyntaxError,
                   TypeError
                ):
                    return set()

            # 4. Create dataframe copy
            database_df = database_df.copy()

            # 5. Create Target

            database_df['Target'] = (
                database_df['scp_codes']
                .apply(
                    lambda codes:
                    int(
                        bool(
                            extract_scp_codes(codes)
                            & mi_codes
                        )
                    )
                )
            )

            logging.info(
                "Target column created successfully"
            )


             # 6. Validate Target

            if not database_df['Target'].isin(
                [0,1]
            ).all():
                raise ValueError(
                    "Target contains values "
                    "other than 0 and 1"
                )

            logging.info(
                'Target validation completed successfully'
            )

            # 7. Target distribution

            target_distribution = (
                database_df['Target']
                .value_counts()
                .sort_index()
            )

            mi_count = int(
                target_distribution.get(1,0)
            )

            non_mi_count = int(
           +     target_distribution.get(0,0)
            )

            total_records = len(
                database_df
            )

            # 8. Validate total records

            if total_records <= 0:
                raise ValueError(
                    "Total records must be greater than zero"
                )

            # 9. Target percentages

            mi_percentage =(
                mi_count / total_records
            ) * 100

            non_mi_percentage = (
                non_mi_count / total_records
            ) * 100

            logging.info(
                f"Total records: {total_records}"
            )

            logging.info(
                f"MI records: {mi_count}"
            )

            logging.info(
                f"Non-MI records: {non_mi_count}"
            )

            logging.info(
                f"MI percentage: "
                f"{mi_percentage:.2f}%"
            )

            logging.info(
                f"Non-MI percentage: "
                f"{non_mi_percentage:.2f}%"
            )

            # 10. Final logging

            logging.info(
                "MI target creation completed successfully"
            )

            return database_df

        except Exception as e:
            logging.exception(
                "Error ocurred during MI target creation"
            )

            raise HealthPulseException(e, sys)
           