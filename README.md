# Unveiling Climate Change Discourse: Unsupervised Sentiment Analysis of Global News Media

![NLP](https://img.shields.io/badge/Domain-NLP-blue) ![Climate](https://img.shields.io/badge/Focus-Climate%20Change-green) ![Status](https://img.shields.io/badge/Status-Research%20Proposal-orange)

## 🎯 Overview
This research quantifies latent media bias in global climate discourse using a multi-faceted unsupervised NLP pipeline. We leverage both lexicon-based models and advanced zero-shot classifiers to gauge sentiment without requiring pre-labeled data. Thematic undercurrents are unearthed using transformer-based topic modeling (BERTopic) to cluster articles by semantic meaning. Bias is then calculated as the sentiment deviation against a dynamic, regional-topical baseline, allowing for robust peer-to-peer comparison. Finally, we employ statistical changepoint detection to identify significant shifts in reporting, correlating them with major world events.

## ✨ Key Highlights
*   **Unsupervised NLP Pipeline:** Reveals media bias in global climate reporting without manual labeling.
*   **Hybrid Sentiment Analysis:** Combines Zero-Shot Transformer inference with VADER sentiment scoring.
*   **Thematic Discovery:** Utilizes **BERTopic** to identify nuanced topics like "Renewable Energy" vs. "Natural Disasters."
*   **Bias Normalization:** Introduces unique baseline scores for fair cross-regional comparisons.
*   **Temporal Analysis:** Identifies event-driven shifts in news tone using **PELT changepoint detection**.

## 🛠️ Technology Stack
*   **Sentiment:** VADER, HuggingFace Zero-Shot Classification (BART/BERT).
*   **Topic Modeling:** BERTopic (Transformer-based embeddings).
*   **Analysis:** `ruptures` (Statistical Changepoint Detection), `pandas`, `scikit-learn`.
*   **Visualization:** `matplotlib`, `seaborn`, `plotly`.

## 📂 Project Structure

The project is organized into a modular pipeline where `main.py` orchestrates the flow from raw data to final visualizations.

```text
climate-news-analysis/
├── data/                       # Raw news articles (aljazeera.jsonl, bbc.jsonl, etc.)
├── src/                        # Source code modules
│   ├── ingest.py               # Loads all articles from data folder
│   ├── preprocess.py           # Cleans text, parses dates, and handles deduplication
│   ├── sentiment.py            # Applies VADER sentiment scoring
│   ├── topics.py               # Implements BERTopic modeling and info extraction
│   ├── aggregate.py            # Groups data by region, time, source, and bias
│   ├── visualize.py            # Generates all PNG plots and timelines
│   ├── reports.py              # Logic for generating text-based analysis reports
│   └── utils.py                # Helper functions for saving/loading data
├── outputs/                    # Processed datasets and visual reports
│   ├── reports/                # Final visual and text outputs
│   │   ├── regional_comparisons/ # Plots comparing climate narratives by region
│   │   ├── source_timelines/     # Sentiment trends for individual news outlets
│   │   ├── bias_report.txt       # Quantified media bias analysis
│   │   └── topic_info.csv        # Metadata for discovered themes
│   ├── final_data.parquet      # Merged dataset with all scores and topics
│   └── processed.parquet       # Intermediate cleaned dataset
├── main.py                     # Entry point to run the entire pipeline
├── requirements.txt            # Project dependencies
└── README.md                   # Documentation
```

## 🚀 Getting Started

1. **Clone the repo:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/climate-discourse-analysis.git
   ```
2. **Install dependencies:**
```bash
pip install -r requirements.txt
```
### Run the pipeline:
The entire research workflow is automated. Run the following command to execute ingestion, sentiment analysis, topic modeling, and visualization in one go:

```bash
python main.py
```

## 👥 Authors
* **Aditya Vasudev K**
* **Ananya Vinay**
