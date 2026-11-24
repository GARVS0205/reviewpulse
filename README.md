# ReviewPulse

**Automatic Opinion Summarization from Customer Reviews**

ReviewPulse analyzes multiple customer reviews to automatically generate explainable summaries of positive opinions, negative opinions, and frequently discussed features. Built with Streamlit, NLTK, and VADER sentiment analysis — requiring no API keys, GPU, external database, or heavy model training.

## Key Features

1. **Overall Sentiment Breakdown**: Classifies opinions into Positive, Neutral, and Negative sentiments using NLTK VADER compound polarity scoring.
2. **General Opinion Summaries**:
   - **What customers liked**: Frequency-ranked, diverse extractive summary quotes for positive opinions.
   - **What customers disliked**: Frequency-ranked, diverse extractive summary quotes for negative opinions.
3. **Frequently Discussed Features Summary**: Automatically detects the most frequently mentioned product features (Battery, Display, Sound, Camera, Performance, Connectivity, Build Quality, Price, Comfort) and extracts specific customer praise and concerns for each feature.
4. **Flexible Review Analysis**:
   - **Preloaded Electronics Dataset**: Explore 20 Amazon Electronics products (150 reviews each, balanced across star ratings) labeled with descriptive names (e.g., HDMI Cables, SanDisk Flash Drives, Kindle Tablets).
   - **Custom Reviews (Text or CSV Upload)**: Type/paste reviews for any custom product or upload any `.csv` file with automatic review-column detection.
5. **Sentiment vs. Star Rating Alignment**: Computes alignment percentage between VADER polarity and numerical star ratings (★4-5 Positive, ★1-2 Negative, ★3 Neutral).
6. **Downloadable Opinion Reports**: Export a complete, formatted text summary report (`.txt`) with one click.
7. **Interactive Review Explorer with Feature Filtering**: Filter reviews by both sentiment label and specific product feature mentions (e.g., Battery, Price, Build Quality).

## Run locally (Windows / macOS / Linux)

### Run directly with Python & Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or using a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The sample CSV and required NLTK resources (`stopwords`, `vader_lexicon`) are pre-bundled in `nltk_data/` for offline startup.

## Understand the code

- `nlp.py`: Explainable NLP pipeline using NLTK: tokenization, stopword filtering with negation retention, Porter stemming, VADER polarity scoring, keyword-based aspect matching, and extractive opinion summarization (`summarize()` and `summarize_features()`).
- `app.py`: Streamlit dashboard with product selection, custom review input, metrics, sentiment and feature charts, opinion summaries, and review browsing.
- `prepare_data.py`: Streaming two-pass sampler that balances star ratings across 20 products from raw review JSONL.
- `data/reviews_sample.csv`: Pre-sampled dataset of 3,000 Amazon Electronics reviews.
- `DEVELOPMENT_LOG.md`: Implementation decisions, verification steps, and project history.

Recreate the sample locally, only if you have the source file:

```powershell
.venv/Scripts/python.exe prepare_data.py --source Electronics_5.json --seed 42
```

Selection deliberately balances star ratings, not VADER labels. Each product has
150 reviews; the sample is not representative of the original rating distribution.
Products are identified only by ASIN because product titles are not available.

## Prepare a GitHub repository and deploy

1. Confirm redistribution terms for the supplied Kaggle source before making
   review text public. The repository is private while the terms remain unverified;
   see `data/README.md`.
2. Create a GitHub repository. Upload only `app.py`, `nlp.py`, `prepare_data.py`,
   `requirements.txt`, `.gitignore`, `.streamlit/config.toml`, `README.md`,
   `DEVELOPMENT_LOG.md`, `data/`, and `nltk_data/`. The legacy audit script/docs
   are optional. Never upload `Electronics_5.json`, `.python/`, `.venv/`, or `.uv-cache/`.
3. In Streamlit Community Cloud, choose **Create app**, connect the repository,
   choose branch `main`, and set main file to `app.py`. If the private repository
   is not listed, grant Streamlit access to that repository through GitHub.
4. In advanced settings, select **Python 3.12**. Deploy. No secrets are required.
5. Select a product and confirm charts and summary quotes appear.

The root requirements file pins the three direct dependencies used during local
verification. Cloud installs their transitive dependencies. Resources are bundled
so application startup does not download NLTK corpora.

Official deployment and dependency instructions:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
and https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies.

## Evaluation talking points

VADER receives original text to preserve casing, punctuation and negation.
Cleaned/stemmed tokens are used for summary relevance. Features are dictionary
matches; sentence sentiment is a simple approximation of feature sentiment.
Summaries copy real sentences and show source line numbers. Ratings are sampling
strata and context, never VADER training labels. There is no gold-standard evaluation
set, so no accuracy claim is made. Sarcasm, mixed opinions and keyword ambiguity
are known limitations.
