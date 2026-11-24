# ReviewPulse development log

## 2026-11-24 — GitHub upload preparation

- User requested creation of a repository and upload of the files required for
  their own Streamlit deployment.
- Confirmed GitHub CLI is authenticated as `GARVS0205`; `GARVS0205/reviewpulse`
  did not already exist.
- User supplied source: https://www.kaggle.com/datasets/shivamparab/amazon-electronics-reviews.
  The page did not expose readable dataset license details during verification.
  Notebook licenses found in search results do not establish the dataset license.
- Selected a private GitHub repository while redistribution terms remain unverified.
  Public exposure of the sample remains subject to confirming those terms.
- Upload scope: app, NLP/sampling code, requirements, configuration, documentation,
  the prepared sample/metadata and bundled NLTK resources. Original 1.5 GB source,
  local environments/caches and one-off verification output are excluded.
- Commands: `git init -b main`, explicit `git add` allowlist, `git diff --cached
  --stat`, `git commit`, `gh repo create GARVS0205/reviewpulse --private`,
  and `git push -u origin main`; remote contents and commit checked after upload.
- Streamlit deployment is left to the user as requested; no browser automation
  or Cloud app creation is part of this upload.

## 2026-11-24 — approved simplified Streamlit implementation

The user approved the electronics dataset and replaced the earlier inspection-only
scope with a single-page Streamlit project. Earlier inspection evidence remains in
`docs/DEVELOPMENT_LOG.md` and `docs/DATASET_SUITABILITY_REPORT.md`.

### Dataset preparation

- Found the unchanged original `Electronics_5.json`, 1,478,965,298 bytes.
- Implemented `prepare_data.py` using standard-library line-by-line `json.loads`.
  It fails with a line number on malformed JSON or non-object records, rather
  than silently treating a JSON array as JSONL or hiding a significant format issue.
- First pass validates fields/dates and counts valid reviews by ASIN and rating.
  Actual result: 1,689,188 records; 1,688,117 valid reviews; 1,071 empty texts.
  Both full passes succeeded. No full-source DataFrame or full-text list is created.
- Eligible products require at least 150 valid reviews and 30 at each star rating.
  There were 102 eligible products; chose the 20 highest valid-review counts,
  breaking ties lexicographically by ASIN.
- Second pass uses seeded reservoir sampling with at most 30 stored reviews per
  product/star bucket (3,000 reviews total). Only selected-product text fingerprints
  are retained for deduplication. This uses memory for counters, fingerprints and
  small reservoirs, rather than the 1.5 GB input. It is not constant memory with
  respect to the number of products or selected-product unique fingerprints.
- Normalized whitespace; removed empty text, invalid ASIN/rating/date records,
  and duplicate casefolded normalized texts across selected products (10 skipped).
- Preserved original case/punctuation and source line references. Dates use ISO format.
  Explicit output whitelist excludes reviewer names and IDs.
- Saved exactly 3,000 reviews, 150 per product, 600 per star overall, to
  `data/reviews_sample.csv`; metadata includes seed 42, selection and CSV checksum.
- Rating balancing improves teaching examples but introduces selection bias. VADER
  sentiment is computed independently; it is not forced to match the star ratings.
- Added `.gitignore` exclusions for original data, JSONL/archives, local Python,
  environments, caches and secrets. Deployment requires only the sample and code.
- User approval permits local development. The earlier unresolved Kaggle source
  URL/license is still unresolved; documented that public redistribution needs
  source-term confirmation. No public upload or deployment was performed.

### NLP implementation

- `nlp.py` keeps the pipeline readable in one class (`ReviewNLP`).
- NLTK RegexpTokenizer lowercases and tokenizes words; English stopwords are
  removed while preserving no/nor/not/never; PorterStemmer creates ranking tokens.
- VADER scores the original unstemmed text, preserving punctuation/case/negation.
  Compound >=0.05 is positive, <=-0.05 negative, otherwise neutral.
- PunktSentenceTokenizer uses untrained punctuation heuristics; this avoids a large
  pretrained tokenizer download but can mis-handle abbreviations.
- Nine regex keyword dictionaries cover battery, display, sound, camera,
  performance, connectivity, build quality, price and comfort.
- A matching feature inherits sentence sentiment. Aggregate unique review/feature/
  sentiment combinations; mixed reviews may contribute to more than one sentiment.
  This is an explainable heuristic, not a trained aspect sentiment model.
- Extractive summaries rank 6–65-word positive/negative sentences using common
  preprocessed tokens, sentiment strength and a small feature-mention bonus.
  Up to three quotes per polarity, one per review; Jaccard token overlap suppresses
  near-identical quotes. Empty candidates show a clear message.
- Quotes retain source line, ASIN, stars and sentence score. No unrelated products
  enter a product's NLP analysis, charts or summaries.
- No model training, external AI service, gold-standard accuracy metric or automated
  test suite. Limitations are visible in the dashboard and README.

### Environment and commands

- Existing system Python was 3.6.2, too old for current dependencies.
- Installed project-local Python 3.12.13 using `uv python install 3.12 --install-dir
  .python --cache-dir .uv-cache`; download required network permission.
- `uv venv --python .python/cpython-3.12.13-windows-x86_64-none/python.exe .venv --cache-dir .uv-cache`
- `uv pip install --python .venv/Scripts/python.exe -r requirements.txt --cache-dir .uv-cache`
- `.venv/Scripts/python.exe prepare_data.py`
- Direct dependencies pinned: Streamlit 1.51.0, pandas 2.3.3, NLTK 3.9.2.
- Streamlit Cloud entrypoint: `app.py`; select Python 3.12; no secrets needed.
  Followed the official Community Cloud dependency/file organization documentation.

### Verification

- Verified the CSV with a one-off local inspection: 3,000 rows, 20 products,
  zero empty texts and zero casefolded duplicate texts, exactly the six whitelisted
  columns. File size: 1,774,397 bytes (about 1.77 MB).
- CSV SHA256: `396003a2b51b6d72a210f43092a5d23b32495fef1a1c1a952a0bb9e15929f3ba`.
- Ran actual `ReviewNLP.analyze` and both summarizers on all 20 ASINs. Every product
  returned feature results and three positive plus three negative quotes.
- Example ASIN B0002L5R78: 150 reviews; 96 positive, 13 neutral, 41 negative;
  eight matching feature groups. Positive quote: “These cables are great for the price.”
- Observed a real limitation: VADER labels “There is no need to pay $30 and up for
  an HDMI cable.” negative, even though its context can express good value. The
  dashboard reports heuristic output, not verified ground truth.
- Downloaded `stopwords` and `vader_lexicon` using `.venv/Scripts/python.exe nlp.py`
  with network permission; bundled in `nltk_data/`. Added a Git ignore exception
  for the resource ZIPs, because VADER reads its lexicon from a ZIP by default.
- Started `.venv/Scripts/python.exe -m streamlit run app.py --server.headless true
  --server.port 8501`. HTTP `/_stcore/health` returned `ok`.
- Browser preview host was unavailable, so visual layout QA could not be completed.
  Used Streamlit's built-in AppTest for a one-off runtime smoke check (no test
  files or automated suite added): initial rendering, changing ASIN to B007WTAJTO,
  and clearing the sentiment filter all completed with zero application exceptions.
  Metrics changed to 83 positive, 16 neutral, 51 negative on the second product.
- Verified quote captions contain original source lines and sentence VADER scores.
- One-off inspection commands were piped through `.venv/Scripts/python.exe -`;
  detailed local output is saved in ignored `verification_output.json`.
- Stopped the verification server after checking it. Run the command in README
  to open the app for evaluation. No automated test suite was created or run.
- Prepared deployment instructions against official Streamlit docs:
  https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
  and https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies.
  Git repository creation, public upload and Cloud deployment were not performed.
