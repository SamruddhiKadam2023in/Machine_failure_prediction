# Stages 1 & 2 — Problem Definition and Dataset Understanding

All numbers in this document were computed by `src/data_understanding.py` from
`data/raw/ai4i2020.csv` (SHA-256 `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`).
Tables are in `reports/stage2/`, figures in `figures/`.

---

## Stage 1 — Problem Definition

**Industrial problem.** Unplanned machine failures in milling/machining stop
production, damage workpieces and tools, and are costly to repair reactively.
Predictive maintenance uses sensor readings to anticipate failures so that
maintenance can be scheduled before breakdown.

**Why failure *type*, not just failure.** Knowing *that* a machine may fail is
less actionable than knowing *how*. Each failure mode calls for a different
response:

| Failure mode | Typical maintenance response |
|---|---|
| TWF – Tool wear failure | Replace the tool |
| HDF – Heat dissipation failure | Check cooling / reduce thermal load |
| PWF – Power failure | Adjust speed–torque operating point |
| OSF – Overstrain failure | Reduce load relative to tool wear and product grade |

**ML objective.** Given one machine's operating snapshot, predict which failure
mode (or no failure) it belongs to.

- **Inputs:** `Type` (L/M/H), Air temperature [K], Process temperature [K],
  Rotational speed [rpm], Torque [Nm], Tool wear [min].
- **Output:** a single failure-type class.
- **Problem type:** supervised, multi-class, heavily imbalanced classification.
- **Primary metric:** Macro F1 (every failure class counts equally, regardless of size).

**Expected outcome.** A validated classifier, chosen by Macro F1 across a baseline,
Random Forest, XGBoost, CatBoost and a stacking ensemble, deployed in a Streamlit
app that returns the predicted failure type and class probabilities.

**Formal objective.** Let

X = (Type, T_air, T_process, ω, τ, t_wear) ∈ {L, M, H} × ℝ⁵

be one machine's operating snapshot and
Y ∈ {No Failure, TWF, HDF, PWF, OSF} its failure type. Learn

**Failure Type = f(X)**, with f: X → Y,

from labelled training pairs (xᵢ, yᵢ), choosing f to maximise the macro-averaged
F1-score on data not used for training. The model also outputs P(Y = k | X) for
each class k, which the app shows as confidence.

**Success criteria.** No numeric targets are set in advance; they would be guesses.
The final model is judged against these criteria, using measured results:

1. **Distinguishes failure classes.** Per-class precision and recall are reported
   for every class, not only overall scores.
2. **Handles minority classes.** Every failure class must be predicted with
   non-trivial recall. A model that wins on accuracy by ignoring a rare class fails
   this criterion.
3. **Strong Macro F1.** The model must clearly beat the Stage 7 baseline on
   Macro F1. This is the primary ranking metric.
4. **Generalises.** Test-set Macro F1 must be consistent with the cross-validation
   estimate, with a small train–validation gap and low variation across folds.
5. **Deployable.** The saved pipeline loads in Streamlit and gives the same
   prediction as the in-memory model for the same input.

**Scope.** Tabular point-in-time prediction on the AI4I 2020 data; offline
training and evaluation; a local Streamlit demo.

**Limitations (known up front).**
- AI4I 2020 is a **synthetic** dataset built to mirror real machining data; its
  failure modes are generated from rule-like conditions, so high scores here do
  not guarantee similar performance on a real shop floor.
- Each row is an independent snapshot; there is no time series, so the model
  predicts the failure state of a snapshot, not time-to-failure.
- Some failure classes have fewer than 100 examples, which limits how reliably
  their per-class metrics can be estimated.

---

## Stage 2 — Dataset Collection & Understanding

### 2.1 Source
UCI Machine Learning Repository, *AI4I 2020 Predictive Maintenance Dataset*
(id 601): https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset.
File supplied by the project author (the workspace could not reach UCI directly).
The CSV begins with a UTF-8 byte-order mark and is read with `encoding="utf-8-sig"`.

### 2.2 Dimensions
**10,000 rows × 14 columns.**

### 2.3 Feature dictionary and data types

| Column | Role | dtype | Unique | Missing |
|---|---|---|---|---|
| UDI | Identifier | int64 | 10000 | 0 |
| Product ID | Identifier | str | 10000 | 0 |
| Type | Categorical feature | str | 3 | 0 |
| Air temperature [K] | Numerical sensor | float64 | 93 | 0 |
| Process temperature [K] | Numerical sensor | float64 | 82 | 0 |
| Rotational speed [rpm] | Numerical sensor | int64 | 941 | 0 |
| Torque [Nm] | Numerical sensor | float64 | 577 | 0 |
| Tool wear [min] | Numerical sensor | int64 | 246 | 0 |
| Machine failure | Binary label | int64 | 2 | 0 |
| TWF / HDF / PWF / OSF / RNF | Failure-mode labels | int64 | 2 each | 0 |

### 2.4 Statistical summary (numerical features)

| Feature | Mean | Std | Min | 25% | Median | 75% | Max | Skew |
|---|---|---|---|---|---|---|---|---|
| Air temperature [K] | 300.005 | 2.000 | 295.3 | 298.3 | 300.1 | 301.5 | 304.5 | 0.114 |
| Process temperature [K] | 310.006 | 1.484 | 305.7 | 308.8 | 310.1 | 311.1 | 313.8 | 0.015 |
| Rotational speed [rpm] | 1538.776 | 179.284 | 1168 | 1423 | 1503 | 1612 | 2886 | 1.993 |
| Torque [Nm] | 39.987 | 9.969 | 3.8 | 33.2 | 40.1 | 46.8 | 76.6 | −0.010 |
| Tool wear [min] | 107.951 | 63.654 | 0 | 53 | 108 | 162 | 253 | 0.027 |

Rotational speed is the only clearly skewed feature (right tail up to 2886 rpm);
outliers are investigated in Stage 3/4 rather than removed now.

**Type:** L 6000 (60.00%), M 2997 (29.97%), H 1003 (10.03%) — `figures/02_product_type_distribution.png`.

### 2.5 Missing values
**0 missing values** in any column.

### 2.6 Duplicates

| Check | Count |
|---|---|
| Exact duplicate rows | 0 |
| Duplicates ignoring UDI and Product ID | 0 |
| Duplicate feature vectors (Type + 5 sensors) | 0 |
| Duplicate UDI / Product ID | 0 / 0 |

### 2.7 Identifier and leakage analysis

| Column | Decision | Reason |
|---|---|---|
| UDI | **Exclude** | Row counter 1..10000 in order. Carries no physical information; any pattern it shows would be an artefact of row ordering. |
| Product ID | **Exclude** | Unique per row (10000 values), so it cannot generalise. Its first letter always equals `Type` (verified for all rows), so the only useful information is already in `Type`. |
| Type | **Keep** | Product quality variant; a genuine operating condition. |
| 5 sensor columns | **Keep** | The physical inputs the model is meant to learn from. |
| Machine failure | **Exclude as feature** | It is a summary of the target (1 whenever a failure mode occurs). Using it as an input is direct target leakage: it would tell the model "failure vs not" for free. |
| TWF, HDF, PWF, OSF, RNF | **Exclude as features** | These *are* the target; they are used only to construct the label. |
| Failure Type | **Not present in this file** | The UCI file has no such column. Some third-party copies (e.g. on Kaggle) add a ready-made `Failure Type` column; here it is *constructed* from the mode flags in Stage 3. As the target, it is never an input. |

### 2.8 Target construction — what the raw labels actually contain

The failure information is stored as one binary flag plus five mode flags,
not as a ready-made multi-class column. Each failure type is represented as follows:

| Failure type | Representation in the raw file |
|---|---|
| No failure | `Machine failure` = 0 and all five mode flags = 0 |
| Tool Wear Failure | `TWF` = 1 |
| Heat Dissipation Failure | `HDF` = 1 |
| Power Failure | `PWF` = 1 |
| Overstrain Failure | `OSF` = 1 |
| Random Failure | `RNF` = 1 |

The mode flags are independent columns, so a row can have more than one set.

| Flag | Count | % of rows |
|---|---|---|
| Machine failure | 339 | 3.39 |
| TWF | 46 | 0.46 |
| HDF | 115 | 1.15 |
| PWF | 95 | 0.95 |
| OSF | 98 | 0.98 |
| RNF | 19 | 0.19 |

Machine failure vs number of mode flags set:

| Machine failure | 0 flags | 1 flag | 2 flags | 3 flags |
|---|---|---|---|---|
| 0 | 9643 | 18 | 0 | 0 |
| 1 | 9 | 306 | 23 | 1 |

Three kinds of rows do not map cleanly to one class:

1. **18 rows: Machine failure = 0 but RNF = 1.** All 18 are RNF; the labels
   contradict each other.
2. **9 rows: Machine failure = 1 but no mode flag.** A failure occurred, but its
   type is not recorded.
3. **24 rows with several mode flags:** PWF+OSF 11, HDF+OSF 6, HDF+PWF 3,
   TWF+OSF 2, TWF+RNF 1, TWF+PWF+OSF 1.

In addition, RNF (random failure) is described by the dataset authors as a
0.1%-probability failure *independent of the process parameters*, so by
construction it cannot be predicted from the inputs. It appears in only 19 rows.

The handling of these cases was approved and is recorded in 2.8.1; it is
implemented in Stage 3 (`src/preprocessing.py`).

### 2.8.1 Final target definition (approved)

**Classes:** No Failure, TWF, HDF, PWF, OSF — five classes.

| Decision | Rows affected | Justification |
|---|---|---|
| RNF is not a target class | 19 rows carry RNF | Described by the dataset authors as independent of the process parameters, so no input can predict it. 18 of its 19 rows also contradict `Machine failure = 0`. It is excluded explicitly, not silently. |
| Exclude RNF=1 with Machine failure=0 | 18 | The two labels contradict each other; neither "No Failure" nor a failure class is trustworthy. |
| Exclude Machine failure=1 with no mode flag | 9 | A failure occurred but its type was not recorded, so no class can be assigned. |
| Exclude rows with more than one of TWF/HDF/PWF/OSF | 23 | A single-label target would need a priority order between modes that the data does not provide. Choosing one would put arbitrary labels into training and into the test set. |
| TWF+RNF row → TWF | 1 | With RNF not a class, this row has exactly one predictable mode. |

**Excluded:** 50 rows (0.50%). **Kept:** 9,950 rows. The excluded rows' UDIs are
saved in Stage 3 so the decision can be audited.

Resulting class distribution (computed in `notebooks/01_data_understanding.ipynb`):

| Class | Count | % |
|---|---|---|
| No Failure | 9643 | 96.91 |
| HDF | 106 | 1.07 |
| PWF | 80 | 0.80 |
| OSF | 78 | 0.78 |
| TWF | 43 | 0.43 |

**Cost of this choice, reported as a limitation:** the 23 multi-mode rows are 7% of
all recorded failures, so the model never sees simultaneous failure modes.
Handling them would need a multi-label formulation, which is out of scope here.

### 2.9 Class imbalance

- Binary view: 9661 no-failure vs 339 failure → **28.5 : 1** (`figures/02_machine_failure_imbalance.png`).
- Mode view: the largest failure mode (HDF, 115) is about 84× smaller than the
  no-failure group; the smallest predictable mode (TWF, 46) is about 210× smaller
  (`figures/02_failure_mode_counts.png`).

Consequences for later stages: accuracy is uninformative (predicting "No Failure"
for everything already scores ~97%), the split and CV must be stratified, and
Macro F1 is the selection metric.
