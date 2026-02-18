import os
import pandas as pd
from src import ingest, preprocess, sentiment, topics, aggregate, visualize, utils, reports

def main():

    #  1. Define Paths 
    DATA_FOLDER = 'data'
    OUTPUT_FOLDER = 'outputs'
    REPORTS_FOLDER = os.path.join(OUTPUT_FOLDER, 'reports')
    
    PROCESSED_DATA_PATH = os.path.join(OUTPUT_FOLDER, 'processed.parquet')
    SENTIMENT_DATA_PATH = os.path.join(OUTPUT_FOLDER, 'sentiments.parquet')
    FINAL_DATA_PATH = os.path.join(OUTPUT_FOLDER, 'final_data.parquet')
    TOPIC_INFO_PATH = os.path.join(REPORTS_FOLDER, 'topic_info.csv')
    TOP_TOPICS_REPORT_PATH = os.path.join(REPORTS_FOLDER, 'top_topics_per_source.txt')
    BIAS_REPORT_PATH = os.path.join(REPORTS_FOLDER, 'bias_report.txt') 
    
    os.makedirs(REPORTS_FOLDER, exist_ok=True)

    #   2. Ingest & Preprocess  
    raw_df = ingest.load_all_articles(DATA_FOLDER)
    processed_df = preprocess.preprocess_data(raw_df)
    utils.save_data(processed_df, PROCESSED_DATA_PATH)
    
    #   3. Sentiment & Topic Modeling  
    # processed_df = utils.load_data(PROCESSED_DATA_PATH)
    sentiment_df = sentiment.apply_vader(processed_df)
    utils.save_data(sentiment_df, SENTIMENT_DATA_PATH)
    
    # sentiment_df = utils.load_data(SENTIMENT_DATA_PATH)
    final_df, topic_model = topics.model_topics(sentiment_df)
    
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(TOPIC_INFO_PATH, index=False)
    utils.save_data(final_df, FINAL_DATA_PATH)

    #   4. Aggregation, Reporting & Visualization  
    print("\n--- Starting Aggregation, Reporting, and Visualization ---")
    final_df = utils.load_data(FINAL_DATA_PATH)
    topic_info_df = pd.read_csv(TOPIC_INFO_PATH)

    # a) Original Aggregations
    agg_time = aggregate.aggregate_by_time(final_df)
    agg_sent_region, agg_topic_region = aggregate.aggregate_by_region(final_df)
    
    # b) New Aggregations & Reports
    agg_sent_source = aggregate.aggregate_by_source(final_df)
    aggregate.generate_top_topics_report(final_df, topic_info_df, TOP_TOPICS_REPORT_PATH)

    bias_analysis_results = aggregate.analyze_bias_and_events(final_df)
    reports.generate_bias_report(bias_analysis_results, BIAS_REPORT_PATH)

    # c) Original Visualizations
    visualize.plot_sentiment_over_time(agg_time, REPORTS_FOLDER)
    visualize.plot_sentiment_by_region(agg_sent_region, REPORTS_FOLDER)
    visualize.plot_topics_by_region(agg_topic_region, REPORTS_FOLDER)

    # d) New Visualizations
    visualize.plot_sentiment_by_source(agg_sent_source, REPORTS_FOLDER)
    visualize.plot_sentiment_over_time_per_source(final_df, REPORTS_FOLDER)
    visualize.plot_regional_comparison_timelines(final_df, REPORTS_FOLDER)

    print("\nPipeline finished successfully!")
    print(f"All outputs and reports have been saved in '{REPORTS_FOLDER}'.")

if __name__ == '__main__':
    main()