from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from tqdm import tqdm

def apply_vader(df, text_column='cleaned_body'):
    """Applies VADER sentiment analysis."""
    print("Applying VADER sentiment analysis...")
    analyzer = SentimentIntensityAnalyzer()
    
    # Use tqdm's progress_apply for a progress bar with pandas
    tqdm.pandas(desc="VADER Progress")
    df['vader_sentiment'] = df[text_column].progress_apply(
        lambda text: analyzer.polarity_scores(text)['compound']
    )
    return df

def apply_zero_shot(df, text_column='cleaned_body', sample_size=None):

    print("Applying Zero-Shot sentiment analysis...")
    if sample_size:
        print(f"Using a sample of {sample_size} articles for Zero-Shot.")
        df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    else:
        df_sample = df

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0) 
    candidate_labels = ['positive', 'negative', 'neutral']
    
    sentiments = []
    for text in tqdm(df_sample[text_column], desc="Zero-Shot Progress"):
        # Truncate text to fit model limits
        truncated_text = text[:1024]
        result = classifier(truncated_text, candidate_labels)
        sentiments.append(result['labels'][0]) # Get the top label
    
    # Add results back to the original dataframe
    df.loc[df_sample.index, 'zero_shot_sentiment'] = sentiments
    return df