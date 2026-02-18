# src/preprocess.py
import pandas as pd
from . import utils

def preprocess_data(df):

    print("Starting preprocessing...")

    # The 'source' column is added during the ingestion step.
    # Drop rows where essential columns are missing.
    df.dropna(subset=['headline', 'body', 'date_published', 'source'], inplace=True)

    # Clean text fields
    df['cleaned_body'] = df['body'].apply(utils.clean_text)
    df['cleaned_headline'] = df['headline'].apply(utils.clean_text)

    # Convert date to datetime objects, coercing errors
    df['date_published'] = pd.to_datetime(df['date_published'], errors='coerce')
    df.dropna(subset=['date_published'], inplace=True)

    # Add country and region information using the new maps from utils
    df['country'] = df['source'].map(utils.SOURCE_TO_COUNTRY_MAP)
    df['region'] = df['country'].map(utils.COUNTRY_TO_REGION_MAP)

    # Fill any potential NaNs if a source isn't in the map (good practice)
    df['country'].fillna('Unknown', inplace=True)
    df['region'].fillna('Unknown', inplace=True)

    # Drop duplicates based on the cleaned article body
    df.drop_duplicates(subset=['cleaned_body'], inplace=True)

    # Filter out articles with very short text
    df = df[df['cleaned_body'].str.len() > 100].copy()

    print(f"Preprocessing complete. {len(df)} articles remaining.")
    return df