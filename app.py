"""Run with: streamlit run app.py"""
from html import escape
from pathlib import Path
import re
import pandas as pd
import streamlit as st
from nlp import ReviewNLP

st.set_page_config(page_title="ReviewPulse - Opinion Summarization", page_icon="💬", layout="wide")

PRODUCT_LABELS = {
    "B0002L5R78": "HDMI Cable (Audio / Video)",
    "B000S5Q9CA": "USB Wall / Phone Charger",
    "B001XURP7W": "SanDisk USB Flash Drive",
    "B002MAPRYU": "Sansa Clip MP3 Player",
    "B002WE6D44": "CompactFlash Camera Memory Card",
    "B003ELYQGG": "In-Ear Stereo Earphones",
    "B0041Q38NU": "Kingston DataTraveler USB Drive",
    "B004QK7HI8": "Mohu Leaf HDTV Antenna",
    "B005CT56F8": "Seagate Expansion External Hard Drive",
    "B005DKZTMG": "Logitech Wireless Keyboard & Mouse",
    "B005FYNSPK": "Mini Micro USB Flash Drive",
    "B005HMKKH4": "External Desktop Hard Drive",
    "B00622AG6S": "High-Speed Car USB Charger",
    "B006GWO5WK": "Amazon PowerFast Kindle Charger",
    "B0074BW614": "Kindle Fire HD 7\" Tablet",
    "B007WTAJTO": "SanDisk Ultra MicroSD Card",
    "B008OHNZI0": "iPhone Anti-Glare Screen Protector",
    "B009SYZ8OC": "Lightning to USB Cable for iPhone",
    "B00BGGDVOO": "Roku HD Streaming Media Player",
    "B00DR0PDNE": "Google Chromecast HDMI Streaming",
}

CUSTOM_PRESETS = {
    "Wireless Bluetooth Headphones": {
        "name": "Sony WH-1000XM5 Wireless Headphones",
        "reviews": """The battery life is phenomenal and easily lasts through a full day of heavy use.
Sound quality is crisp with punchy bass, though the microphone volume is a bit low.
Terrible build quality; the plastic headband snapped after just two weeks of normal use.
Super comfortable fit for long flights and bluetooth pairing is instantaneous with zero lag.
Way too expensive for what you get, especially with the flimsy carrying case."""
    },
    "Smart 4K Ultra HD TV": {
        "name": "Samsung 55-inch 4K Crystal UHD TV",
        "reviews": """The OLED screen display is absolutely breathtaking with rich colors and deep blacks.
Built-in sound speakers are surprisingly loud and clear for a thin television.
Slow interface performance with frustrating lag when switching between streaming apps.
Great value for the price compared to other premium brand monitors.
Remote control feels cheap and the Wi-Fi connectivity drops frequently."""
    },
    "Ergonomic Mechanical Keyboard": {
        "name": "Keychron K2 Mechanical Keyboard",
        "reviews": """Incredible typing comfort and ergonomic wrist support for 8-hour workdays.
Build quality is outstanding with heavy aluminum construction and durable switches.
High price tag, but the performance and tactile response are totally worth it.
Wireless bluetooth connectivity keeps lagging and disconnecting from my laptop.
The battery dies within three days if the RGB backlighting is on maximum."""
    },
    "Custom / Type Your Own": {
        "name": "My Custom Electronics Item",
        "reviews": """The battery life is phenomenal and charges quickly.
Screen display is sharp, bright, and vibrant outdoors.
Poor build quality with fragile plastic that cracks easily.
Affordable price and great overall value for everyday use."""
    }
}


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


def render_opinion_quote(item, sentiment):
    border_color = "#167D8D" if sentiment == "Positive" else "#C2410C"
    icon = "👍" if sentiment == "Positive" else "👎"
    bg_color = "#F0F5F7" if sentiment == "Positive" else "#FEF2F2"
    sentence = escape(item["sentence"])
    source_line = escape(str(item.get("source_line", "N/A")))
    feat_tag = f" · <i>Feature: {escape(', '.join(item['features']))}</i>" if item.get("features") else ""
    stars_tag = f"{escape(str(item.get('overall', 0)))}★ · " if item.get("overall", 0) > 0 else ""
    st.markdown(
        f"""
        <div style="border-left: 4px solid {border_color}; padding: 10px 14px; margin-bottom: 12px; background-color: {bg_color}; border-radius: 4px;">
            <div style="font-size: 0.95rem; color: #162D40; line-height: 1.4;">{icon} "{sentence}"</div>
            <div style="font-size: 0.8rem; color: #64748B; margin-top: 6px;">
                Line {source_line} · {stars_tag}VADER: <b>{item['compound']:+.2f}</b>{feat_tag}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def generate_summary_report(product_name, identifier, reviews, counts, pos_summary, neg_summary, feat_summaries, agreement_info=None):
    total = max(len(reviews), 1)
    lines = [
        "=" * 68,
        "REVIEWPULSE — AUTOMATIC OPINION SUMMARIZATION REPORT",
        "=" * 68,
        f"Product / Item: {product_name}",
        f"Identifier: {identifier}",
        f"Total Reviews Analyzed: {len(reviews)}",
        f"Sentiment Breakdown: {counts['Positive']} Positive ({counts['Positive']/total*100:.1f}%), "
        f"{counts['Neutral']} Neutral ({counts['Neutral']/total*100:.1f}%), "
        f"{counts['Negative']} Negative ({counts['Negative']/total*100:.1f}%)",
    ]
    if agreement_info:
        lines.append(f"Star Rating vs Sentiment Alignment: {agreement_info}")

    lines.extend([
        "-" * 68,
        "\n1. WHAT CUSTOMERS LIKED (POSITIVE OPINIONS):",
    ])
    if pos_summary:
        for i, q in enumerate(pos_summary, 1):
            feat_info = f" [Aspect: {', '.join(q['features'])}]" if q.get("features") else ""
            lines.append(f"  {i}. \"{q['sentence']}\" (Score: {q['compound']:+.2f}){feat_info}")
    else:
        lines.append("  No prominent positive quotes found.")

    lines.extend([
        "\n2. WHAT CUSTOMERS DISLIKED (NEGATIVE OPINIONS):",
    ])
    if neg_summary:
        for i, q in enumerate(neg_summary, 1):
            feat_info = f" [Aspect: {', '.join(q['features'])}]" if q.get("features") else ""
            lines.append(f"  {i}. \"{q['sentence']}\" (Score: {q['compound']:+.2f}){feat_info}")
    else:
        lines.append("  No prominent negative quotes found.")

    lines.extend([
        "\n3. FREQUENTLY DISCUSSED FEATURES SUMMARY:",
    ])
    if feat_summaries:
        for f in feat_summaries:
            lines.append(f"\n  • Feature: {f['feature']} ({f['total_mentions']} total mentions)")
            lines.append(f"    Sentiment: {f['pos_ratio']:.0f}% Positive, {f['neg_ratio']:.0f}% Negative, {100 - f['pos_ratio'] - f['neg_ratio']:.0f}% Neutral")
            if f.get("positive_quote"):
                lines.append(f"    Praise: \"{f['positive_quote']['sentence']}\" (Score: {f['positive_quote']['compound']:+.2f})")
            if f.get("negative_quote"):
                lines.append(f"    Concern: \"{f['negative_quote']['sentence']}\" (Score: {f['negative_quote']['compound']:+.2f})")
    else:
        lines.append("  No specific feature discussions detected.")

    lines.extend([
        "\n" + "=" * 68,
        "Generated by ReviewPulse (NLTK Tokenizer + Porter Stemmer + VADER Sentiment Analyzer)",
        "=" * 68,
    ])
    return "\n".join(lines)


st.title("ReviewPulse")
st.write("Automatic Opinion Summarization from Customer Reviews: analyze opinions, sentiment, and frequently discussed features.")
st.caption("Explainable NLP · NLTK + VADER Sentiment · Extractive Opinion Summarization")

try:
    data = load_reviews()
    nlp = get_nlp()
except (OSError, LookupError) as exc:
    st.error("Required sample or NLTK resources are missing. Follow README.md setup instructions.")
    st.code(str(exc))
    st.stop()

# Mode selection: Preloaded sample products vs custom reviews
mode = st.radio(
    "Choose Review Source",
    ["Preloaded Product Dataset (20 Electronics ASINs)", "Analyze Custom Reviews (Paste / Enter Text / Upload CSV)"],
    horizontal=True,
)

agreement_info = None

if mode == "Preloaded Product Dataset (20 Electronics ASINs)":
    asin = st.selectbox(
        "Select a product (ASIN)",
        sorted(data["asin"].unique()),
        format_func=lambda x: f"{x} — {PRODUCT_LABELS.get(x, 'Electronics Product')}",
    )
    product_item_name = PRODUCT_LABELS.get(asin, "Electronics Product")
    product_identifier = f"ASIN: {asin}"
    product = data[data["asin"] == asin].copy()
    product["product"] = product_item_name
    reviews, features, sentences = analyze_product(product)

    # Compute Sentiment vs. Star Rating agreement
    if "overall" in reviews.columns and (reviews["overall"] > 0).any():
        star_polarity = reviews["overall"].map(lambda s: "Positive" if s >= 4 else "Negative" if s <= 2 else "Neutral")
        matches = int((reviews["sentiment"] == star_polarity).sum())
        total = len(reviews)
        pct = (matches / total) * 100
        agreement_info = f"{pct:.1f}% ({matches}/{total} reviews aligned with numerical star rating polarity)"

else:
    input_method = st.radio(
        "Custom Input Method:",
        ["📝 Type / Paste Reviews", "📁 Upload CSV File"],
        horizontal=True,
    )

    if input_method == "📝 Type / Paste Reviews":
        col_preset, col_prod = st.columns([2, 3])
        with col_preset:
            preset_key = st.selectbox(
                "Load sample product reviews or enter your own:",
                list(CUSTOM_PRESETS.keys()),
                index=0,
                help="Pick a sample product template or select 'Custom / Type Your Own'.",
            )
        with col_prod:
            custom_product_name = st.text_input(
                "Product / Item Name:",
                value=CUSTOM_PRESETS[preset_key]["name"],
                placeholder="e.g. Sony WH-1000XM5, iPhone 15, Kindle Oasis...",
                help="Specify the name of the product or item these reviews are about.",
            )

        product_item_name = custom_product_name.strip() if custom_product_name.strip() else "Custom Product"
        product_identifier = "Custom Customer Reviews"

        custom_text = st.text_area(
            f"Enter customer reviews for '{product_item_name}' (one review per line):",
            value=CUSTOM_PRESETS[preset_key]["reviews"],
            height=140,
            help="Paste multiple reviews here to analyze sentiment and extract opinion summaries on the fly.",
        )
        lines = [line.strip() for line in custom_text.strip().split("\n") if line.strip()]
        if not lines:
            st.warning("Please enter at least one review text.")
            st.stop()
        custom_df = pd.DataFrame({
            "product": product_item_name,
            "reviewText": lines,
            "asin": "Custom",
            "overall": 0,
            "summary": [line[:40] + ("..." if len(line) > 40 else "") for line in lines],
            "source_line": list(range(1, len(lines) + 1)),
            "reviewTime": "N/A",
        })
        reviews, features, sentences = nlp.analyze(custom_df)

    else:
        # CSV File Upload
        uploaded_file = st.file_uploader(
            "Upload a CSV file containing customer reviews:",
            type=["csv"],
            help="Upload any CSV file. The app will detect the review text column and summarize opinions.",
        )
        if uploaded_file is None:
            st.info("Please upload a `.csv` file to begin analysis.")
            st.stop()

        try:
            raw_csv = pd.read_csv(uploaded_file, keep_default_na=False)
        except Exception as exc:
            st.error(f"Error reading CSV: {exc}")
            st.stop()

        if raw_csv.empty:
            st.warning("The uploaded CSV is empty.")
            st.stop()

        # Find best candidate for review text column
        candidate_cols = [c for c in raw_csv.columns if any(k in c.lower() for k in ["review", "text", "comment", "feedback", "body", "content"])]
        default_col_idx = raw_csv.columns.get_loc(candidate_cols[0]) if candidate_cols else 0

        col_col, col_pname = st.columns([1, 1])
        with col_col:
            selected_text_col = st.selectbox(
                "Select column containing review text:",
                raw_csv.columns.tolist(),
                index=int(default_col_idx),
            )
        with col_pname:
            file_suggested_name = Path(uploaded_file.name).stem.replace("_", " ").title()
            custom_product_name = st.text_input(
                "Product / Item Name:",
                value=file_suggested_name,
                help="Name of the product or item in this CSV.",
            )

        product_item_name = custom_product_name.strip() if custom_product_name.strip() else file_suggested_name
        product_identifier = f"Uploaded CSV: {uploaded_file.name}"

        # Clean review rows
        valid_texts = raw_csv[selected_text_col].astype(str).str.strip()
        valid_mask = valid_texts.str.len() > 0
        filtered_csv = raw_csv[valid_mask].copy()

        if filtered_csv.empty:
            st.warning("No valid text entries found in the selected column.")
            st.stop()

        filtered_csv["product"] = product_item_name
        filtered_csv["reviewText"] = valid_texts[valid_mask]
        filtered_csv["asin"] = "Custom-CSV"
        filtered_csv["overall"] = filtered_csv["overall"] if "overall" in filtered_csv.columns else 0
        filtered_csv["source_line"] = list(range(1, len(filtered_csv) + 1))
        filtered_csv["summary"] = filtered_csv["summary"] if "summary" in filtered_csv.columns else filtered_csv["reviewText"].str.slice(0, 40)

        # Analyze
        with st.spinner(f"Analyzing {len(filtered_csv)} reviews..."):
            reviews, features, sentences = nlp.analyze(filtered_csv)

# Prominent Product Item Badge & Actions
col_badge, col_dl = st.columns([4, 1])
with col_badge:
    st.markdown(
        f"""
        <div style="background: linear-gradient(90deg, #167D8D 0%, #0E5A66 100%); padding: 14px 20px; border-radius: 8px; margin: 15px 0 10px 0; color: white;">
            <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.85;">Product Item Under Analysis</div>
            <div style="font-size: 1.35rem; font-weight: 700; margin-top: 2px;">📦 {escape(product_item_name)}</div>
            <div style="font-size: 0.85rem; opacity: 0.9; margin-top: 4px;"><b>{escape(product_identifier)}</b> &nbsp;·&nbsp; {len(reviews)} reviews analyzed</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Overall Sentiment Metrics
counts = reviews["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Reviews analyzed", len(reviews))
m2.metric("Positive opinions", int(counts["Positive"]), f"{counts['Positive']/len(reviews)*100:.0f}%" if len(reviews) else None)
m3.metric("Neutral opinions", int(counts["Neutral"]), f"{counts['Neutral']/len(reviews)*100:.0f}%" if len(reviews) else None)
m4.metric("Negative opinions", int(counts["Negative"]), f"{counts['Negative']/len(reviews)*100:.0f}%" if len(reviews) else None)

# Agreement Metric Callout if applicable
if agreement_info:
    st.caption(f"🎯 **Sentiment vs. Star Rating Polarity Agreement:** **{agreement_info}** (★4–5 mapped Positive, ★1–2 Negative, ★3 Neutral)")

# Charts: Overall sentiment and Feature opinions
left, right = st.columns(2)
with left:
    st.subheader(f"Sentiment Breakdown · {product_item_name}")
    st.bar_chart(counts.rename("Reviews"), color="#167D8D")
with right:
    st.subheader(f"Feature Mentions · {product_item_name}")
    if features.empty:
        st.write("No configured feature keywords were detected in these reviews.")
    else:
        st.bar_chart(features, stack=True)
    st.caption("Reviews mentioning a feature, classified by sentence sentiment. Reviews with multiple mentions count across features.")

st.divider()

# Core Deliverable 1 & 2: Summaries of Positive and Negative Opinions
st.subheader(f"General Opinion Summaries — {product_item_name}")
st.caption("Representative sentences extracted and ranked by token frequency relevance, sentiment intensity, and feature evidence.")
positive, negative = st.columns(2)

pos_summary = nlp.summarize(sentences, "Positive", limit=3)
neg_summary = nlp.summarize(sentences, "Negative", limit=3)

with positive:
    st.markdown("### What customers liked (Positive Opinions)")
    if not pos_summary:
        st.info("Not enough suitable positive sentences to summarize.")
    for item in pos_summary:
        render_opinion_quote(item, "Positive")

with negative:
    st.markdown("### What customers disliked (Negative Opinions)")
    if not neg_summary:
        st.info("Not enough suitable negative sentences to summarize.")
    for item in neg_summary:
        render_opinion_quote(item, "Negative")

st.divider()

# Core Deliverable 3: Summaries of Frequently Discussed Features
st.subheader(f"Frequently Discussed Features Summary — {product_item_name}")
st.caption("Detailed opinion summaries for the features most frequently discussed by customers.")

feature_summaries = nlp.summarize_features(sentences, features, top_n=3)
if not feature_summaries:
    st.info("No specific product features had enough discussion to generate feature summaries.")
else:
    f_cols = st.columns(len(feature_summaries))
    for col, feat_info in zip(f_cols, feature_summaries):
        with col:
            st.markdown(f"### {feat_info['feature']}")
            st.markdown(
                f"**Total Mentions:** {feat_info['total_mentions']} &nbsp;|&nbsp; "
                f"<span style='color:#167D8D'><b>{feat_info['pos_ratio']:.0f}% Positive</b></span> &nbsp;|&nbsp; "
                f"<span style='color:#C2410C'><b>{feat_info['neg_ratio']:.0f}% Negative</b></span>",
                unsafe_allow_html=True,
            )
            st.write("")
            if feat_info["positive_quote"]:
                st.markdown("**👍 Customer Praise:**")
                st.write(f"\"{feat_info['positive_quote']['sentence']}\"")
                st.caption(f"Score: {feat_info['positive_quote']['compound']:+.2f}")
            else:
                st.caption("No strong positive quote found for this feature.")

            if feat_info["negative_quote"]:
                st.markdown("**👎 Customer Concern:**")
                st.write(f"\"{feat_info['negative_quote']['sentence']}\"")
                st.caption(f"Score: {feat_info['negative_quote']['compound']:+.2f}")
            else:
                st.caption("No strong negative quote found for this feature.")

# Download Report Button
st.divider()
safe_filename = re.sub(r"[^A-Za-z0-9_-]", "_", product_item_name.lower())
summary_report_text = generate_summary_report(
    product_item_name, product_identifier, reviews, counts, pos_summary, neg_summary, feature_summaries, agreement_info
)
st.download_button(
    label="📥 Download Opinion Summary Report (.txt)",
    data=summary_report_text,
    file_name=f"opinion_summary_{safe_filename}.txt",
    mime="text/plain",
    help="Download a clean text report summarizing positive opinions, negative opinions, and frequently discussed features.",
)

# Review Exploration & Details
with st.expander(f"Browse raw reviews for {product_item_name}"):
    filter_c1, filter_c2 = st.columns([1, 1])
    with filter_c1:
        selected_sentiments = st.multiselect("Filter reviews by VADER label", list(counts.index), default=list(counts.index))
    with filter_c2:
        available_features = ["All Features"] + sorted(features.index.tolist()) if not features.empty else ["All Features"]
        selected_feature = st.selectbox("Filter reviews by feature mention:", available_features)

    shown = reviews[reviews["sentiment"].isin(selected_sentiments)]
    if selected_feature != "All Features" and "features_mentioned" in shown.columns:
        shown = shown[shown["features_mentioned"].apply(lambda feats: selected_feature in feats)]

    st.caption(f"Showing **{len(shown)}** of **{len(reviews)}** reviews matching filters.")

    display_df = shown.copy()
    if "features_mentioned" in display_df.columns:
        display_df["features"] = display_df["features_mentioned"].apply(lambda f: ", ".join(f) if f else "None")

    display_cols = [c for c in ["product", "features", "source_line", "reviewTime", "overall", "summary", "reviewText", "sentiment", "compound"] if c in display_df.columns]
    st.dataframe(display_df[display_cols], hide_index=True, width="stretch")
    if not shown.empty and "processed_tokens" in shown.columns:
        st.write("Sample processed tokens (used for summary ranking):")
        st.code(" ".join(shown.iloc[0]["processed_tokens"][:80]))

with st.expander("How it works and NLP methodology"):
    st.write("1. **Preprocessing**: NLTK tokenization, punctuation removal, English stopword removal (while preserving negation like 'no', 'not', 'never'), and Porter stemming.")
    st.write("2. **Sentiment Analysis**: NLTK VADER analyzer scores original text to preserve punctuation, capitalization, and negation cues. Scores >= 0.05 are classified Positive, <= -0.05 Negative, and others Neutral.")
    st.write("3. **Feature Detection**: Heuristic keyword dictionaries match 9 common electronics aspects (Battery, Display, Sound, Camera, Performance, Connectivity, Build quality, Price, Comfort).")
    st.write("4. **Opinion Summarization**: Extractive summarization algorithm ranks candidate sentences by word frequency relevance, sentiment intensity, and feature evidence, with Jaccard diversity filtering to avoid repetitive quotes.")
    st.write("5. **Frequently Discussed Features**: Ranks features by customer mention volume and generates balanced positive/negative opinion summaries for top aspects.")
