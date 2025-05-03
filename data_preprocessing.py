import pandas as pd
import re

class DataPreprocessor:
    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        """
        Lowercase, remove special characters, and extra spaces from text.
        """
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with null values and clean all string columns.
        """
        df = df.dropna()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].apply(self.clean_text)
        return df

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Example feature extraction: add text length column for all string columns.
        """
        for col in df.select_dtypes(include=['object']).columns:
            df[f'{col}_length'] = df[col].apply(len)
        return df 