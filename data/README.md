# Dataset notes

The user approved use of the supplied Kaggle electronics review dataset on
2026-11-24. The local source is `Electronics_5.json` (1,478,965,298 bytes).
Both complete streaming passes verified UTF-8 JSONL, with 1,689,188 objects and
1,071 blank review texts. No source review is loaded into a full-data DataFrame.

The sample contains 3,000 unique normalized review texts, 20 ASINs, and exactly
30 reviews per star per ASIN. The 20 highest-count eligible products were selected;
102 products met the minimum coverage rule. Ten duplicate-text occurrences among
selected products were skipped. See `reviews_sample.metadata.json` for product
IDs, cleaning counts, seed, selection rule, and CSV SHA256.

Columns: `asin`, `reviewText`, `overall`, `summary` (customer headline),
`reviewTime` (ISO date), `source_line` (1-based line in original file).
Reviewer names and IDs are excluded by an explicit field whitelist. Free-text
reviews are preserved; this is not a guarantee that reviewers never mentioned
personal information in their prose. Text duplicates use whitespace normalization
and case folding, with SHA256 fingerprints. Near-duplicate paraphrases are not removed.

The filename, schema and count are consistent with Amazon Reviews 2014,
Electronics 5-core: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html.
That consistency does not prove this Kaggle copy's provenance or license.
The user supplied the Kaggle source URL:
https://www.kaggle.com/datasets/shivamparab/amazon-electronics-reviews.
The dataset page did not expose readable license information during verification.
No data license is asserted by this project. Confirm those terms before public GitHub upload or
Community Cloud deployment; a small subset does not automatically become redistributable.

Suggested source attribution: Amazon Reviews 2014, collected by Julian McAuley
and collaborators. Consult the original source page for citation details and terms.
This dataset is historical, English-oriented, and deliberately rating-balanced;
its charts do not estimate market-wide or product-wide satisfaction.
