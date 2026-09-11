# Two-Tower Retrieval on Amazon-Electronics

A PyTorch two-tower (DSSM-style) model trained on the Amazon Product Review
dataset. User-side features are aggregated from behaviour history with strict
temporal causality; item-side features come from the product metadata dump.

## Data

Download the two Amazon-Electronics dumps and place them in one directory:

```
reviews_Electronics_5.json.gz    # 5-core reviews, ~1.69M interactions
meta_Electronics.json.gz         # product metadata, ~498K items
```

## Quick start

```bash
pip install -r requirements.txt

# smoke test on generated mock data (no download needed)
python -m src.tools.make_mock_data --out-dir data/mock
python -m src.train --raw-dir data/mock --sample-users 0 --epochs 5 --device cpu

# real run: 10k sampled users
python -m src.train --raw-dir /path/to/amazon --sample-users 10000

# full dataset
python -m src.train --raw-dir /path/to/amazon --sample-users 0
```

Parsed tables are cached under `data/processed/`. Re-running skips parsing and
feature building unless you pass `--force`.

### CLI options

| flag | meaning |
|---|---|
| `--raw-dir` | directory holding the two `.json.gz` files |
| `--sample-users N` | subsample N users; `0` means all |
| `--epochs` / `--batch-size` / `--lr` | training overrides |
| `--loss bce\|softmax` | pointwise BCE (default) or in-batch softmax |
| `--device cpu\|cuda\|mps` | defaults to cuda → mps → cpu |
| `--force` | ignore cached tables and rebuild |

## Pipeline

```
parse_amazon.py  reviews + meta .json.gz  ->  interactions / items tables
features.py      per-user chronological walk  ->  leak-free user features
encoders.py      vocabs + scalers + item table (fitted on train split only)
dataset.py       negative sampling, train / valid / test loaders
model/towers.py  UserTower, ItemTower, dot-product scoring
train.py         training loop, early stopping, final test report
evaluate.py      AUC, GAUC, logloss, Recall@K, NDCG@K, MRR
```

## Leak prevention

This is the part of the design that matters most, and it is enforced in three
places:

1. **Feature construction.** `features.py` walks each user's events in
   chronological order and emits the feature row *before* folding the current
   event into the accumulator. There is no code path that can read the present
   or the future.
2. **Fitting statistics.** Normalisation constants, item popularity and item
   average rating are computed on the training split only, then applied
   unchanged to valid and test.
3. **Splitting.** Leave-one-out per user: newest event → test, second newest →
   valid, everything earlier → train. Guaranteed chronological.

Verify with:

```bash
python -m src.tools.check_leakage --raw-dir data/mock
```

It recomputes features with an independent naive implementation and asserts the
results match, checks that no behaviour sequence contains its own target or any
later item, and checks split chronology per user.

## User features

Twenty-eight dense features plus three embedding groups, all derived from
events strictly preceding the current one.

| group | features |
|---|---|
| counts / ratings | `hist_len`, `avg_rating`, `std_rating`, `pos_ratio`, `last_rating` |
| engagement | `avg_review_len`, `avg_summary_len`, `avg_helpful_ratio`, `helpful_missing_ratio` |
| temporal | `days_since_last`, `lifetime_span`, `avg_gap`, `recent_{7,30,90}d_cnt` |
| price affinity | `avg_price`, `max_price`, `std_price`, `price_missing_ratio` |
| diversity | `n_unique_cat`, `n_unique_brand`, `cat_entropy`, `cat_repeat_ratio` |
| calendar | `dow_sin/cos`, `month_sin/cos` |
| embeddings | `user_id`, behaviour sequence (last 20 items, masked mean-pool), `top_cat`, `top_brand` |

Counts and durations are `log1p`-compressed before z-scoring, since both are
heavy-tailed.

**No user × item crossing features.** They would be strong predictors, but a
tower that sees the target item cannot have its output precomputed, which
defeats the purpose of the architecture. The constraint is structural: the
towers share no inputs and meet only at the final dot product.

## Item features

`item_id`, `brand`, `cat_l1`, `cat_leaf` as embeddings; `price`, `sales_rank`,
`title_len`, `n_also_bought`, `n_also_viewed`, `item_pop`, `item_avg_rating`,
`item_pos_ratio` as dense. Price and sales rank are missing for a large
fraction of items, so each carries an explicit missing indicator rather than
being silently imputed.

## Model

```
user features ──> UserTower (MLP 256→128→64) ──> L2 norm ──┐
                                                            ├──> dot / τ + b ──> BCE
item features ──> ItemTower (MLP 256→128→64) ──> L2 norm ──┘
```

- **Temperature** is learnable and parameterised in log space, clamped to
  `[0.01, 1.0]`. Normalised dot products live in `[-1, 1]`, far too narrow for
  BCE to express confident probabilities; dividing by τ widens the range.
- **Global bias** lets the model calibrate the base positive rate, which a
  normalised dot product alone cannot represent.
- **Dense features** pass through a small MLP before concatenation so that ~28
  scalars are not drowned out by ~190 dimensions of embeddings.
- **Sequence pooling** is masked mean-pooling. Target-attention (DIN-style)
  would break tower independence and is deliberately not used.

## Sampling and evaluation

Training draws 4 negatives per positive from a `popularity^0.75` distribution,
resampled every epoch, excluding items the user has interacted with. Evaluation
uses a frozen 1-positive + 99-negative candidate set so metrics stay comparable
across epochs.

Reported metrics: **AUC**, **GAUC** (per-user AUC weighted by group size),
**logloss**, plus Recall@{10,20,50}, NDCG@{10,20,50} and MRR. GAUC is usually
the more meaningful number here, since it removes differences in per-user
rating baselines.

Results are written to `artifacts/report.json` and the best checkpoint to
`artifacts/best_model.pt`.
