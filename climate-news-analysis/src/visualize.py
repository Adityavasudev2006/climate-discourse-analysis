# src/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd


def plot_sentiment_over_time(df_time, output_folder):
    print("Generating overall sentiment over time plot...")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_time, x='date_published', y='vader_sentiment')
    plt.title('Average Climate News Sentiment Over Time (All Sources)')
    plt.xlabel('Date')
    plt.ylabel('Average VADER Sentiment Score')
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'sentiment_over_time_ALL.png'))
    plt.close()

def plot_sentiment_by_region(df_region, output_folder):
    print("Generating sentiment by region plot...")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_region, x='region', y='vader_sentiment', palette='viridis')
    plt.title('Average Climate News Sentiment by Region')
    plt.xlabel('Region')
    plt.ylabel('Average VADER Sentiment Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'sentiment_by_region.png'))
    plt.close()

def plot_topics_by_region(df_topics, output_folder):
    print("Generating topic distribution plot...")
    df_topics.plot(
        kind='bar',
        stacked=True,
        figsize=(14, 8),
        colormap='tab20'
    ).legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Topic Distribution by News Region')
    plt.xlabel('Region')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'topics_by_region.png'))
    plt.close()


def plot_sentiment_by_source(df_source, output_folder):
    print("Generating sentiment by source plot...")
    plt.figure(figsize=(12, 10))
    sns.barplot(data=df_source, x='vader_sentiment', y='source', palette='coolwarm_r', orient='h')
    plt.title('Average Climate News Sentiment by Source', fontsize=16)
    plt.xlabel('Average VADER Sentiment Score', fontsize=12)
    plt.ylabel('News Source', fontsize=12)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8) # Add a line for neutral
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'sentiment_by_source.png'))
    plt.close()

def plot_sentiment_over_time_per_source(df, output_folder):
    print("Generating individual sentiment timelines for each source...")
    source_timelines_folder = os.path.join(output_folder, 'source_timelines')
    os.makedirs(source_timelines_folder, exist_ok=True)

    all_sources = df['source'].unique()
    
    for source in all_sources:
        plt.figure(figsize=(12, 6))
        source_df = df[df['source'] == source].copy()
        
        timeline = source_df.set_index('date_published')['vader_sentiment'].resample('3M').mean()
        
        if timeline.dropna().empty:
            print(f"Skipping plot for {source} due to insufficient data.")
            plt.close()
            continue

        timeline.plot(kind='line', marker='.', linestyle='-')
        plt.title(f'Sentiment Over Time: {source.title()}', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Avg VADER Sentiment (3-Month Rolling)', fontsize=12)
        plt.ylim(-1, 1) # Keep y-axis consistent across all plots
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        plt.savefig(os.path.join(source_timelines_folder, f'{source}_timeline.png'))
        plt.close()
    print(f"Individual timelines saved to '{source_timelines_folder}'.")

def plot_regional_comparison_timelines(df, output_folder):
    print("Generating regional comparison sentiment timelines...")
    regional_folder = os.path.join(output_folder, 'regional_comparisons')
    os.makedirs(regional_folder, exist_ok=True)
    
    source_counts_by_region = df.groupby('region')['source'].nunique()
    regions_to_compare = source_counts_by_region[source_counts_by_region > 1].index.tolist()

    for region in regions_to_compare:
        plt.figure(figsize=(14, 7))
        region_df = df[df['region'] == region]
        
        
        pivot_df = region_df.pivot_table(
            index=pd.Grouper(key='date_published', freq='3M'), # Quarterly aggregation
            columns='source',
            values='vader_sentiment',
            aggfunc='mean'
        )

        if pivot_df.dropna(how='all').empty:
            print(f"Skipping comparison plot for {region} due to insufficient data.")
            plt.close()
            continue
            
        pivot_df.plot(ax=plt.gca(), marker='o', linestyle='--', markersize=4)
        
        plt.title(f'Sentiment Comparison in {region}', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Avg VADER Sentiment (3-Month Rolling)', fontsize=12)
        plt.ylim(-1, 1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(title='Source')
        plt.tight_layout()
        
        plt.savefig(os.path.join(regional_folder, f'{region}_comparison.png'))
        plt.close()
    print(f"Regional comparisons saved to '{regional_folder}'.")