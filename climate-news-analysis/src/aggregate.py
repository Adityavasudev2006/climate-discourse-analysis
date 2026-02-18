# src/aggregate.py
import pandas as pd
import numpy as np
import ruptures as rpt
from . import utils

def aggregate_by_time(df, freq='M'):
    print(f"Aggregating data by time frequency: {freq}")
    df_time = df.set_index('date_published').copy()
    sentiment_over_time = df_time['vader_sentiment'].resample(freq).mean().reset_index()
    return sentiment_over_time

def aggregate_by_region(df):
    print("Aggregating data by region...")
    sentiment_by_region = df.groupby('region')['vader_sentiment'].mean().reset_index()
    topic_by_region = df.groupby(['region', 'topic']).size().unstack(fill_value=0)
    return sentiment_by_region, topic_by_region

def aggregate_by_source(df):
    print("Aggregating data by news source...")
    sentiment_by_source = df.groupby('source')['vader_sentiment'].mean().sort_values(ascending=False).reset_index()
    return sentiment_by_source

def generate_top_topics_report(df, topic_info_df, output_path):
    print(f"Generating top topics report at {output_path}...")
    
    # Merge topic names into the main dataframe for easier access
    topic_names = topic_info_df[['Topic', 'Name']].set_index('Topic')
    df_merged = df.join(topic_names, on='topic')
    # Use a placeholder for topics without a generated name (like Topic -1)
    df_merged['Name'].fillna(df_merged['topic'].apply(lambda x: f"Topic {x}"), inplace=True)

    all_sources = sorted(df['source'].unique())
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Top 50 Most Frequent Topics per News Source\n")
        f.write("===============================================\n\n")

        for source in all_sources:
            f.write(f"--- {source.upper()} ---\n")
            
            source_df = df_merged[df_merged['source'] == source]
            topic_counts = source_df['Name'].value_counts().nlargest(50)
            
            if topic_counts.empty:
                f.write("No topic data available.\n\n")
                continue

            for i, (topic_name, count) in enumerate(topic_counts.items()):
                f.write(f"{i+1:2}. {topic_name} ({count} articles)\n")
            
            f.write("\n")
    print("Report generation complete.")

def analyze_bias_and_events(df):

    print("\n--- Starting Bias and Event Correlation Analysis ---")
    
    # 1. Calculate Regional Baseline Sentiment per Topic per Month
    print("Step 1/4: Calculating regional baseline sentiment...")
    df['month'] = df['date_published'].dt.to_period('M')
    baseline_sentiment = df.groupby(['region', 'topic', 'month'])['vader_sentiment'].transform('mean')
    df['baseline_sentiment'] = baseline_sentiment
    
    # 2. Calculate Bias Score for each article
    print("Step 2/4: Calculating bias scores...")
    df['bias_score'] = df['vader_sentiment'] - df['baseline_sentiment']
    df.dropna(subset=['bias_score'], inplace=True) # Drop articles without a baseline

    # 3. Analyze each source for changepoints
    print("Step 3/4: Detecting sentiment changepoints for each source...")
    all_sources = df['source'].unique()
    analysis_results = {}

    # Convert event strings to datetime objects
    events_dt = {
        name: (pd.to_datetime(dates[0]), pd.to_datetime(dates[1]))
        for name, dates in utils.MAJOR_EVENTS.items()
    }

    for source in all_sources:
        source_df = df[df['source'] == source].copy()
        
        # Resample bias score weekly to find trends. Fill missing weeks.
        weekly_bias = source_df.set_index('date_published')['bias_score'].resample('W').mean().fillna(0)
        
        if len(weekly_bias) < 10: # Not enough data to analyze
            continue
            
        # Changepoint detection using the Ruptures library
        points = weekly_bias.values.reshape(-1, 1)
        algo = rpt.Pelt(model="rbf").fit(points)
        # We'll consider up to 5 major changepoints per source
        result_indices = algo.predict(pen=3) 

        changepoints = []
        if len(result_indices) > 1:
            # The last index is the end of the series, so we ignore it
            for cp_idx in result_indices[:-1]:
                changepoint_date = weekly_bias.index[cp_idx]
                correlated_event = "None"
                
                # Check if this changepoint happened within 90 days AFTER a major event started
                for event, (start_date, end_date) in events_dt.items():
                    if start_date <= changepoint_date <= start_date + pd.Timedelta(days=90):
                        correlated_event = event
                        break
                
                changepoints.append({
                    "date": changepoint_date,
                    "correlated_event": correlated_event
                })

        analysis_results[source] = {
            "overall_bias_score": source_df['bias_score'].mean(),
            "changepoints": changepoints
        }
        
    print("Step 4/4: Finalizing analysis...")
    print("Bias and event analysis complete.")
    return analysis_results