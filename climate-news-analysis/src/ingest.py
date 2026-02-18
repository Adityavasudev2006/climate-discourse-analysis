# src/ingest.py
import pandas as pd
import os
import json
from tqdm import tqdm

def load_all_articles(data_folder_path):

    all_articles = []
    print(f"Ingesting articles from {data_folder_path}...")

    # Use tqdm for a progress bar
    for filename in tqdm(os.listdir(data_folder_path), desc="Reading files"):
        if filename.endswith('.jsonl'):
            
            
            # Get the source name from the filename (e.g., 'bbc.jsonl' -> 'bbc')
            source_name = filename.replace('.jsonl', '')
            
            file_path = os.path.join(data_folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        article = json.loads(line)
                        
                        # Add the source name to the article's data before appending
                        article['source'] = source_name
                        
                        all_articles.append(article)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode line in {filename}")

    if not all_articles:
        raise ValueError("No articles were loaded. Check the data directory and file format.")

    df = pd.DataFrame(all_articles)
    print(f"Ingested {len(df)} total articles.")
    return df