"""Two streaming passes; only product counts and bounded review pools stay in memory."""
import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

FIELDS = ["asin", "reviewText", "overall", "summary", "reviewTime", "source_line"]


def records(path, stats):
    """Validate actual JSONL structure and expose only approved application fields."""
    with open(path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            stats["lines"] += 1
            if not line.strip():
                stats["blank_lines"] += 1
                continue
            try:
                item = json.loads(line)
            except ValueError as exc:
                raise ValueError("Invalid JSONL at line {}: {}".format(line_number, exc))
            if not isinstance(item, dict):
                raise ValueError("Expected one JSON object per line, line {}".format(line_number))
            text = item.get("reviewText")
            asin = item.get("asin")
            rating = item.get("overall")
            if not isinstance(text, str) or not text.strip():
                stats["empty_reviews"] += 1
                continue
            if not isinstance(asin, str) or not asin.strip() or isinstance(rating, bool) or rating not in (1, 2, 3, 4, 5):
                stats["invalid_product_or_rating"] += 1
                continue
            try:
                date = datetime.strptime(item.get("reviewTime", ""), "%m %d, %Y").date().isoformat()
            except (ValueError, TypeError):
                stats["invalid_dates"] += 1
                continue
            stats["valid_reviews"] += 1
            yield dict(asin=asin.strip(), reviewText=re.sub(r"\s+", " ", text).strip(),
                       overall=int(rating), summary=re.sub(r"\s+", " ", str(item.get("summary") or "")).strip(),
                       reviewTime=date, source_line=line_number)


def prepare(source, output, seed=42):
    # Pass 1: counts are small (one counter per ASIN), never a full review table.
    stats = Counter()
    counts = defaultdict(Counter)
    for row in records(source, stats):
        counts[row["asin"]][row["overall"]] += 1
    # Require enough reviews in every rating to support a useful demonstration.
    eligible = [asin for asin, ratings in counts.items()
                if sum(ratings.values()) >= 150 and min(ratings[s] for s in range(1, 6)) >= 30]
    eligible.sort(key=lambda asin: (-sum(counts[asin].values()), asin))
    products = eligible[:20]
    if len(products) < 20:
        raise ValueError("Fewer than 20 products have adequate rating coverage; inspect before changing selection.")
    print("Pass 1: {} valid reviews; {} eligible products. Sampling 20.".format(stats["valid_reviews"], len(eligible)), flush=True)

    # Pass 2: reservoir sampling keeps at most 30 reviews per product/star bucket.
    # Hash sets store only fingerprints for selected products, never source texts.
    rng = random.Random(seed)
    pools = defaultdict(list)
    seen_counts = Counter()
    seen_text = set()
    duplicate_count = 0
    second_pass = Counter()
    selected = set(products)
    for row in records(source, second_pass):
        if row["asin"] not in selected:
            continue
        fingerprint = hashlib.sha256(row["reviewText"].casefold().encode("utf-8")).digest()
        if fingerprint in seen_text:
            duplicate_count += 1
            continue
        seen_text.add(fingerprint)
        key = (row["asin"], row["overall"])
        seen_counts[key] += 1
        bucket = pools[key]
        if len(bucket) < 30:
            bucket.append(row)
        else:
            index = rng.randrange(seen_counts[key])
            if index < 30:
                bucket[index] = row
    rows = [row for key in sorted(pools) for row in pools[key]]
    if len(rows) != 3000:
        raise ValueError("Deduplication left {} reviews; inspect coverage before proceeding.".format(len(rows)))
    rows.sort(key=lambda row: (row["asin"], row["source_line"]))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = {"source_filename": Path(source).name, "source_bytes": Path(source).stat().st_size,
              "format": "UTF-8 JSON Lines, verified on both complete passes", "seed": seed,
              "pass_one": dict(stats), "selected_products": products,
              "selection": "20 highest-count ASINs with >=150 valid reviews and >=30 per star; 30 unique texts per star per product",
              "duplicates_removed_in_selected_products": duplicate_count,
              "duplicate_key": "SHA256 of casefolded, whitespace-normalized review text, across selected products",
              "rows": len(rows), "ratings": dict(Counter(row["overall"] for row in rows)),
              "csv_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(".metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="Electronics_5.json")
    parser.add_argument("--output", default="data/reviews_sample.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.source, args.output, args.seed)
