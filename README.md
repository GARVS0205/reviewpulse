# ReviewPulse

A small university NLP project: select one electronics product, inspect VADER
sentiment and keyword feature opinions, and read extractive positive/negative quotes.
No API key, database, GPU, or model training is required.

## Run locally (Windows PowerShell)

Use Python **3.12**, not the older system Python 3.6. The prepared environment in
this workspace already uses Python 3.12.

```powershell
.venv/Scripts/python.exe -m streamlit run app.py
```

For a fresh checkout with Python 3.12 installed:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m streamlit run app.py
```

The CSV and small NLTK resources are included. You do not need the large source
file to run or deploy the app. To refresh NLTK resources, run
`.venv/Scripts/python.exe nlp.py` (requires internet).

## Understand the code

- `prepare_data.py`: validates JSONL one line at a time in two passes, selects
  20 well-reviewed ASINs, deduplicates texts and samples 30 reviews per star per
  product with seed 42. Saves 3,000 reviews and preparation metadata.
- `nlp.py`: NLTK tokenization, stopword removal and stemming; VADER scores;
  feature keyword matches; frequency-ranked, diverse source-sentence summaries.
- `app.py`: one Streamlit page; caches the small CSV, NLP resources and product analysis.
- `data/reviews_sample.csv`: application data, with no reviewer name or ID columns.
- `DEVELOPMENT_LOG.md`: decisions, commands, verification and limitations.

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
