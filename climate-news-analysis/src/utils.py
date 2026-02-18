# src/utils.py
import pandas as pd
import os
import re

# Detailed mapping from source filename (without .jsonl) to its country
SOURCE_TO_COUNTRY_MAP = {
    'aljazeera': 'Qatar',
    'atlantic': 'USA',
    'batimes': 'Argentina',
    'bbc': 'UK',
    'china_daily': 'China',
    'cnn': 'USA',
    'daily_nation': 'Kenya',
    'daily_post': 'Nigeria',
    'dailymail': 'UK',
    'dw': 'Germany',
    'economist': 'UK',
    'folha': 'Brazil',
    'fox': 'USA',
    'guardian': 'UK',
    'independent': 'UK',
    'newshub': 'New Zealand',
    'nytimes': 'USA',
    'nzherald': 'New Zealand',
    'skyau': 'Australia',
    'stuff': 'New Zealand',
    'washington_post': 'USA',
    'yomiuri': 'Japan'
}

# Mapping from country to a broader geographical region for summary charts
COUNTRY_TO_REGION_MAP = {
    'Qatar': 'Middle East',
    'USA': 'North America',
    'Argentina': 'South America',
    'UK': 'Europe',
    'China': 'Asia',
    'Kenya': 'Africa',
    'Nigeria': 'Africa',
    'Germany': 'Europe',
    'Brazil': 'South America',
    'New Zealand': 'Oceania',
    'Australia': 'Oceania',
    'Japan': 'Asia'
}

MAJOR_EVENTS = {
    "Syrian Civil War Escalation (Chemical Weapons)": ("2013-08-21", "2013-10-01"),
    "Westgate Shopping Mall Attack, Kenya": ("2013-09-21", "2013-09-24"),
    "Typhoon Haiyan in the Philippines": ("2013-11-08", "2013-11-15"),
    "Euromaidan Revolution in Ukraine": ("2013-11-21", "2014-02-22"),
    "Annexation of Crimea by Russia": ("2014-02-23", "2014-03-19"),
    "MH370 Disappearance": ("2014-03-08", "2014-04-28"),
    "Boko Haram Chibok Kidnapping, Nigeria": ("2014-04-14", "2014-05-01"),
    "MH17 Shot Down over Ukraine": ("2014-07-17", "2014-07-25"),
    "Ebola Outbreak in West Africa (Peak Fear)": ("2014-08-01", "2014-12-31"),
    "Rise of ISIS in Iraq and Syria": ("2014-06-01", "2014-12-31"),
    "Hong Kong Umbrella Revolution": ("2014-09-26", "2014-12-15"),
    "Charlie Hebdo Attack in Paris": ("2015-01-07", "2015-01-14"),
    "Germanwings Flight 9525 Crash": ("2015-03-24", "2015-04-01"),
    "Nepal Earthquake": ("2015-04-25", "2015-05-12"),
    "Greek Debt Crisis (Bailout Referendum)": ("2015-06-27", "2015-07-13"),
    "Iran Nuclear Deal Agreement": ("2015-07-14", "2015-07-20"),
    "European Migrant Crisis (Peak)": ("2015-08-01", "2015-11-30"),
    "Paris Bataclan Attacks": ("2015-11-13", "2015-11-20"),
    "COP21 Paris Climate Agreement": ("2015-11-30", "2015-12-12"),
}

def clean_text(text):
    """A simple text cleaning function."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def save_data(df, path):
    """Saves a DataFrame to a Parquet file."""
    print(f"Saving data to {path}...")
    df.to_parquet(path, index=False)
    print("Save complete.")

def load_data(path):
    """Loads a DataFrame from a Parquet file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at {path}")
    print(f"Loading data from {path}...")
    df = pd.read_parquet(path)
    print("Load complete.")
    return df