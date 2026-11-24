"""Small, explainable NLP pipeline. Original text is preserved for VADER/quotes."""
from collections import Counter
from pathlib import Path
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer
from nltk.tokenize import PunktSentenceTokenizer, RegexpTokenizer

DATA_DIR = Path(__file__).parent / "nltk_data"
nltk.data.path.insert(0, str(DATA_DIR))
FEATURES = {
    "Battery": r"\bbatter(?:y|ies)\b",
    "Display": r"\b(?:screen|display|monitor)\b",
    "Sound": r"\b(?:sound|audio|bass|volume)\b",
    "Camera": r"\b(?:camera|photos?|pictures?)\b",
    "Performance": r"\b(?:performance|speed|fast|slow)\b",
    "Connectivity": r"\b(?:wi-?fi|bluetooth|connection|connectivity)\b",
    "Build quality": r"\b(?:durable|durability|build quality|plastic|sturdy)\b",
    "Price": r"\b(?:price|cost|expensive|cheap|value)\b",
    "Comfort": r"\b(?:comfort|comfortable|uncomfortable)\b",
}
PATTERNS = {name: re.compile(pattern, re.I) for name, pattern in FEATURES.items()}


def setup_resources():
    """Run locally once; bundled resources make Cloud startup network independent."""
    DATA_DIR.mkdir(exist_ok=True)
    for resource in ("stopwords", "vader_lexicon"):
        if not nltk.download(resource, download_dir=str(DATA_DIR), quiet=True, raise_on_error=True):
            raise RuntimeError("Could not download " + resource)


class ReviewNLP:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        # Keep negation; removing it changes the meaning of customer opinions.
        self.stop_words = set(stopwords.words("english")) - {"no", "nor", "not", "never"}
        self.stemmer = PorterStemmer()
        self.words = RegexpTokenizer(r"[A-Za-z]+(?:'[A-Za-z]+)?")
        # Untrained Punkt uses punctuation heuristics; no large model download needed.
        self.sentences = PunktSentenceTokenizer()

    def preprocess(self, text):
        return [self.stemmer.stem(word) for word in self.words.tokenize(text.lower())
                if word not in self.stop_words and len(word) > 1]

    def score(self, text):
        return self.analyzer.polarity_scores(text)["compound"]

    @staticmethod
    def label(score):
        return "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"

    def analyze(self, reviews):
        """Only the chosen product enters this method."""
        reviews = reviews.copy()
        reviews["compound"] = reviews["reviewText"].map(self.score)
        reviews["sentiment"] = reviews["compound"].map(self.label)
        reviews["processed_tokens"] = reviews["reviewText"].map(self.preprocess)
        sentence_rows = []
        feature_rows = []
        for row in reviews.to_dict("records"):
            for sentence in self.sentences.tokenize(row["reviewText"]):
                score = self.score(sentence)
                features = [name for name, pattern in PATTERNS.items() if pattern.search(sentence)]
                item = {"sentence": sentence, "compound": score, "sentiment": self.label(score),
                        "source_line": row["source_line"], "asin": row["asin"], "overall": row["overall"],
                        "features": features, "tokens": self.preprocess(sentence)}
                sentence_rows.append(item)
                for feature in features:
                    feature_rows.append({"Feature": feature, "Sentiment": item["sentiment"],
                                         "source_line": row["source_line"]})
        features = pd.DataFrame(feature_rows, columns=["Feature", "Sentiment", "source_line"])
        # Count each review once per feature/sentiment, even with repeated mentions.
        features = features.drop_duplicates()
        if features.empty:
            table = pd.DataFrame(columns=["Positive", "Neutral", "Negative"])
        else:
            table = pd.crosstab(features["Feature"], features["Sentiment"]).reindex(
                columns=["Positive", "Neutral", "Negative"], fill_value=0)
        return reviews, table, sentence_rows

    def summarize(self, sentences, sentiment, limit=3):
        """Rank actual sentences by token frequency, sentiment, and feature evidence."""
        candidates = [s for s in sentences if s["sentiment"] == sentiment
                      and 6 <= len(self.words.tokenize(s["sentence"])) <= 65]
        frequency = Counter(token for s in candidates for token in s["tokens"])
        def rank(s):
            terms = s["tokens"]
            relevance = sum(frequency[t] for t in set(terms)) / max(len(terms), 1)
            return relevance * (1 + abs(s["compound"])) * (1.2 if s["features"] else 1)
        candidates.sort(key=lambda s: (-rank(s), s["source_line"], s["sentence"]))
        chosen = []
        used_reviews = set()
        for sentence in candidates:
            terms = set(sentence["tokens"])
            if sentence["source_line"] in used_reviews:
                continue
            # Avoid near-identical quotes using token-set Jaccard similarity.
            if any(len(terms & set(other["tokens"])) / max(len(terms | set(other["tokens"])), 1) > 0.6
                   for other in chosen):
                continue
            chosen.append(sentence)
            used_reviews.add(sentence["source_line"])
            if len(chosen) == limit:
                break
        return chosen


if __name__ == "__main__":
    setup_resources()
    print("NLTK stopwords and VADER lexicon ready in nltk_data/.")
