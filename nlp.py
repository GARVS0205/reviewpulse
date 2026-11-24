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
    "Battery": r"\b(?:batter(?:y|ies)|charg(?:e|er|ing|ed))\b",
    "Display": r"\b(?:screen|display|monitor|resolution)\b",
    "Sound": r"\b(?:sound|audio|bass|volume|speakers?|mic(?:rophone)?)\b",
    "Camera": r"\b(?:camera|photos?|pictures?|lens|video)\b",
    "Performance": r"\b(?:performance|speed|fast|slow|lag|responsive)\b",
    "Connectivity": r"\b(?:wi-?fi|bluetooth|connection|connectivity|pairing)\b",
    "Build quality": r"\b(?:durable|durability|build quality|plastic|sturdy|flimsy|broke|broken)\b",
    "Price": r"\b(?:price|cost|expensive|cheap|value|worth|affordable)\b",
    "Comfort": r"\b(?:comfort|comfortable|uncomfortable|fit)\b",
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
        for idx, row in enumerate(reviews.to_dict("records")):
            source_line = row.get("source_line", idx + 1)
            asin = row.get("asin", "N/A")
            overall = row.get("overall", 0)
            text = str(row.get("reviewText", ""))
            for sentence in self.sentences.tokenize(text):
                score = self.score(sentence)
                features = [name for name, pattern in PATTERNS.items() if pattern.search(sentence)]
                item = {"sentence": sentence, "compound": score, "sentiment": self.label(score),
                        "source_line": source_line, "asin": asin, "overall": overall,
                        "features": features, "tokens": self.preprocess(sentence)}
                sentence_rows.append(item)
                for feature in features:
                    feature_rows.append({"Feature": feature, "Sentiment": item["sentiment"],
                                         "source_line": source_line})
        features = pd.DataFrame(feature_rows, columns=["Feature", "Sentiment", "source_line"])
        # Count each review once per feature/sentiment, even with repeated mentions.
        features = features.drop_duplicates()
        if features.empty:
            table = pd.DataFrame(columns=["Positive", "Neutral", "Negative"])
        else:
            table = pd.crosstab(features["Feature"], features["Sentiment"]).reindex(
                columns=["Positive", "Neutral", "Negative"], fill_value=0)

        # Map mentioned features back to each review row for filtering and inspection
        review_features_map = {}
        for item in sentence_rows:
            line_key = item["source_line"]
            if line_key not in review_features_map:
                review_features_map[line_key] = set()
            review_features_map[line_key].update(item["features"])

        reviews["features_mentioned"] = [
            sorted(review_features_map.get(row.get("source_line", idx + 1), set()))
            for idx, row in enumerate(reviews.to_dict("records"))
        ]
        return reviews, table, sentence_rows

    def summarize(self, sentences, sentiment, limit=3, min_words=6):
        """Rank actual sentences by token frequency, sentiment, and feature evidence."""
        candidates = [s for s in sentences if s["sentiment"] == sentiment
                      and min_words <= len(self.words.tokenize(s["sentence"])) <= 65]
        # Fallback to shorter sentences if not enough candidates meet the standard length filter
        if len(candidates) < limit and min_words > 3:
            candidates = [s for s in sentences if s["sentiment"] == sentiment
                          and 3 <= len(self.words.tokenize(s["sentence"])) <= 65]
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

    def summarize_features(self, sentences, features_table, top_n=3):
        """Generate structured opinion summaries for the most frequently discussed features."""
        if features_table.empty:
            return []
        table = features_table.copy()
        table["Total"] = table["Positive"] + table["Neutral"] + table["Negative"]
        top_features = table[table["Total"] > 0].sort_values(by="Total", ascending=False).head(top_n)
        summaries = []
        for feature, row in top_features.iterrows():
            total = int(row["Total"])
            pos_count = int(row["Positive"])
            neg_count = int(row["Negative"])
            neu_count = int(row["Neutral"])
            feat_sentences = [s for s in sentences if feature in s["features"]]
            pos_quotes = self.summarize(feat_sentences, "Positive", limit=1, min_words=4)
            neg_quotes = self.summarize(feat_sentences, "Negative", limit=1, min_words=4)
            pos_ratio = (pos_count / total * 100) if total > 0 else 0
            neg_ratio = (neg_count / total * 100) if total > 0 else 0
            summaries.append({
                "feature": feature,
                "total_mentions": total,
                "positive_count": pos_count,
                "negative_count": neg_count,
                "neutral_count": neu_count,
                "pos_ratio": pos_ratio,
                "neg_ratio": neg_ratio,
                "positive_quote": pos_quotes[0] if pos_quotes else None,
                "negative_quote": neg_quotes[0] if neg_quotes else None,
            })
        return summaries


if __name__ == "__main__":
    setup_resources()
    print("NLTK stopwords and VADER lexicon ready in nltk_data/.")
