"""Generate tiny fake Amazon-format dumps so the pipeline can be smoke-tested.

The mock mimics the real quirks: reviews are strict JSON lines, metadata is
Python-repr with single quotes, prices and sales ranks are often missing.

    python -m src.tools.make_mock_data --out-dir data/mock
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

BRANDS = ["Sony", "Canon", "Anker", "Logitech", "Bose", "Dell", "HP", ""]
CATS = [
    ["Electronics", "Camera & Photo", "Digital Cameras"],
    ["Electronics", "Computers", "Laptops"],
    ["Electronics", "Accessories", "Cables"],
    ["Electronics", "Audio", "Headphones"],
    ["Electronics", "Television", "LED TVs"],
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=str, default="data/mock")
    parser.add_argument("--users", type=int, default=300)
    parser.add_argument("--items", type=int, default=400)
    parser.add_argument("--min-events", type=int, default=6)
    parser.add_argument("--max-events", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = ["B{:08d}".format(i) for i in range(args.items)]
    item_cat = {a: CATS[rng.integers(len(CATS))] for a in items}
    item_brand = {a: BRANDS[rng.integers(len(BRANDS))] for a in items}
    # Zipf-ish popularity so negative sampling has something to chew on.
    pop = 1.0 / np.arange(1, args.items + 1) ** 0.8
    pop = pop / pop.sum()

    reviews_path = out_dir / "reviews_Electronics_5.json.gz"
    with gzip.open(reviews_path, "wt", encoding="utf-8") as f:
        for u in range(args.users):
            uid = "A{:09d}".format(u)
            n_events = int(rng.integers(args.min_events, args.max_events))
            picked = rng.choice(args.items, size=n_events, replace=False, p=pop)
            ts = int(rng.integers(1_100_000_000, 1_300_000_000))
            # Each user has a taste bias, giving the model real signal to learn.
            bias = rng.normal(0.0, 0.6)
            for asin_idx in picked:
                ts += int(rng.integers(1, 60)) * 86400
                rating = float(np.clip(round(rng.normal(4.0 + bias, 1.0)), 1, 5))
                total_votes = int(rng.integers(0, 20))
                useful = int(rng.integers(0, total_votes + 1))
                f.write(
                    json.dumps(
                        {
                            "reviewerID": uid,
                            "asin": items[asin_idx],
                            "overall": rating,
                            "unixReviewTime": ts,
                            "reviewText": "word " * int(rng.integers(5, 200)),
                            "summary": "good " * int(rng.integers(1, 8)),
                            "helpful": [useful, total_votes],
                        }
                    )
                    + "\n"
                )

    meta_path = out_dir / "meta_Electronics.json.gz"
    with gzip.open(meta_path, "wt", encoding="utf-8") as f:
        for asin in items:
            record = {
                "asin": asin,
                "title": "Product " + asin + " great value",
                "categories": [item_cat[asin]],
                "related": {
                    "also_bought": list(rng.choice(items, size=int(rng.integers(0, 10)))),
                    "also_viewed": list(rng.choice(items, size=int(rng.integers(0, 10)))),
                },
            }
            if rng.random() > 0.4:  # ~40% missing, like the real dump
                record["price"] = round(float(rng.uniform(5, 900)), 2)
            if rng.random() > 0.3:
                record["salesRank"] = {"Electronics": int(rng.integers(1, 500000))}
            if item_brand[asin]:
                record["brand"] = item_brand[asin]
            # Deliberately written as a Python repr, exactly like the real file.
            f.write(repr(record) + "\n")

    print("wrote {} and {}".format(reviews_path, meta_path))


if __name__ == "__main__":
    main()
