"""Run with: streamlit run app.py"""
from pathlib import Path
import pandas as pd
import streamlit as st
from nlp import ReviewNLP

st.set_page_config(page_title="ReviewPulse", page_icon="💬", layout="wide")


@st.cache_data
def load_reviews():
    # Load only the small prepared CSV. The original JSON is never read by the app.
    return pd.read_csv(Path(__file__).parent / "data/reviews_sample.csv", keep_default_na=False,
                       dtype={"asin": str})


@st.cache_resource
def get_nlp():
    return ReviewNLP()


@st.cache_data(max_entries=20)
def analyze_product(product_reviews):
    return get_nlp().analyze(product_reviews)


st.title("ReviewPulse")
st.write("Explore customer opinions, product features, and the reasons behind the sentiment.")
st.caption("Historical electronics reviews · NLTK + VADER · Extractive opinion summaries")
try:
    data = load_reviews()
    nlp = get_nlp()
except (OSError, LookupError) as exc:
    st.error("Required sample or NLTK resources are missing. Follow README.md setup instructions.")
    st.code(str(exc))
    st.stop()

asin = st.selectbox("Select a product (ASIN)", sorted(data["asin"].unique()))
st.caption("Product titles are unavailable in the source. Every chart and summary below uses only this ASIN.")
product = data[data["asin"] == asin]
reviews, features, sentences = analyze_product(product)
counts = reviews["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
a, b, c, d = st.columns(4)
a.metric("Reviews analyzed", len(reviews))
b.metric("Positive (VADER)", int(counts["Positive"]))
c.metric("Neutral (VADER)", int(counts["Neutral"]))
d.metric("Negative (VADER)", int(counts["Negative"]))
st.info("This demo uses 3,000 reviews across 20 products, balanced by star rating (30 per star per product). "
        "It is a teaching sample, not an estimate of overall customer satisfaction. VADER labels can differ from stars.")

left, right = st.columns(2)
with left:
    st.subheader("Review sentiment")
    st.bar_chart(counts.rename("Reviews"), color="#167D8D")
with right:
    st.subheader("Feature opinions")
    if features.empty:
        st.write("No configured feature keywords were found for this product.")
    else:
        st.bar_chart(features, stack=True)
    st.caption("Reviews mentioning a feature, labeled using sentence sentiment. A mixed review can count in multiple sentiments; features can overlap.")

st.subheader("Opinion summaries")
st.caption("Selected original sentences, ranked by common words, sentiment strength, and feature mentions. These are quotes, not generated claims.")
positive, negative = st.columns(2)
for column, sentiment, title in [(positive, "Positive", "What customers liked"),
                                  (negative, "Negative", "What customers disliked")]:
    with column:
        st.markdown("### " + title)
        summary = nlp.summarize(sentences, sentiment)
        if not summary:
            st.write("Not enough suitable {} sentences to summarize.".format(sentiment.lower()))
        for item in summary:
            st.write(item["sentence"])
            st.caption("Source line {} · {} stars · VADER {:+.2f}".format(
                item["source_line"], item["overall"], item["compound"]))

with st.expander("Browse reviews and preprocessing"):
    selected_sentiments = st.multiselect("Filter reviews by VADER label", list(counts.index), default=list(counts.index))
    shown = reviews[reviews["sentiment"].isin(selected_sentiments)]
    st.dataframe(shown[["source_line", "reviewTime", "overall", "summary", "reviewText", "sentiment", "compound"]],
                 hide_index=True, width="stretch")
    if not shown.empty:
        st.write("Example processed tokens (used for summary ranking):")
        st.code(" ".join(shown.iloc[0]["processed_tokens"][:80]))

with st.expander("How it works and limitations"):
    st.write("NLTK tokenizes, removes English stopwords while keeping negation, and applies Porter stemming. "
             "VADER scores original text so punctuation, capitalization and negation remain available. "
             "Compound scores ≥ 0.05 are positive, ≤ −0.05 negative, and otherwise neutral.")
    st.write("Features use a small keyword dictionary. A feature inherits its sentence's sentiment; this is "
             "a heuristic, not a trained aspect model. Mixed sentences, sarcasm, domain terms, and sentence "
             "boundaries can cause errors. Summary quotes may omit context. No accuracy benchmark is claimed.")
    st.write("Source: supplied Kaggle Electronics_5.json, consistent with Amazon Reviews 2014 Electronics 5-core. "
             "Reviews are historical; reviewer name and ID fields are excluded. See data/README.md for provenance and redistribution status.")
