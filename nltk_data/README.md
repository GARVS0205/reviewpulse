# Bundled NLTK resources

Downloaded using NLTK 3.9.2's official downloader on 2026-11-24:
`stopwords` and `vader_lexicon`. Kept in the standard directory structure so
the application starts without downloading data from the network.

`corpora/stopwords/README` contains upstream stopword source information.
VADER source and license: https://github.com/cjhutto/vaderSentiment
NLTK data catalog: https://www.nltk.org/nltk_data/

Refresh locally with `python nlp.py`, then commit the updated resources as well
as any code changes. The `.gitignore` exception for resource ZIPs is required.
