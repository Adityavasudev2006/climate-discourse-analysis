from bertopic import BERTopic
import pandas as pd

def model_topics(df, text_column='cleaned_body'):
    print("Starting topic modeling with BERTopic...")
    # BERTopic can be slow. Consider using a GPU-accelerated UMAP if available.
    topic_model = BERTopic(verbose=True, calculate_probabilities=False)
    
    # Ensure text column is a list of strings
    docs = df[text_column].tolist()
    
    topics, _ = topic_model.fit_transform(docs)
    
    df['topic'] = topics
    
    print("Topic modeling complete.")
    return df, topic_model