import pandas as pd


class CleanData:
    @staticmethod
    def input_empty_values(df: pd.DataFrame):
        num_cols = df.select_dtypes(
            include="number"
        ).columns
        for col in num_cols:
            if df[col].isnull().any():
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
        return df

    @staticmethod
    def clean_col(name: str):
        return (
            name.strip().lower().replace(" ", "_")
        )