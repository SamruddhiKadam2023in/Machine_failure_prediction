# Project Completion Checklist

Status of every stage in the project brief. Each item points to where its output lives. Every result was
regenerated from a clean state with `python run_pipeline.py` (log: `reports/pipeline_run_log.txt`) and
matched the earlier run exactly.

| # | Item | Status | Where |
|---|---|---|---|
| 0 | Project setup, environment, pinned requirements | ✅ | `requirements.txt`, `config.py` |
| 1 | Problem definition | ✅ | `reports/01_02_problem_definition_and_data_understanding.md`, report ch. 2 |
| 2 | Dataset collection | ✅ | `data/raw/ai4i2020.csv` (SHA-256 in `models/model_metadata.json`) |
| 3 | Dataset understanding | ✅ | `src/data_understanding.py`, `notebooks/01_data_understanding.ipynb`, `reports/stage2/` |
| 4 | Data cleaning | ✅ | `src/preprocessing.py`, `reports/stage3/excluded_rows.csv` (50 rows, each with a reason) |
| 5 | Preprocessing pipeline | ✅ | `build_preprocessor()` in `src/preprocessing.py`, `notebooks/03_preprocessing.ipynb` |
| 6 | EDA | ✅ | `src/eda.py`, `notebooks/02_eda.ipynb`, `figures/04_*.png` |
| 7 | Feature engineering | ✅ | `src/feature_engineering.py` (3 physical features) |
| 8 | Feature selection | ✅ | `src/feature_study.py`, `reports/stage5/feature_set_comparison.csv` |
| 9 | Train–test split (stratified 80/20, seed 42) | ✅ | `data/processed/train.csv`, `test.csv`, `reports/stage5/split_summary.csv` |
| 10 | Baseline model | ✅ | Dummy + Logistic Regression, `notebooks/05_baseline.ipynb` |
| 11 | Random Forest (bagging) | ✅ | `notebooks/06_random_forest.ipynb` |
| 12 | XGBoost (boosting) | ✅ | `notebooks/07_xgboost.ipynb` |
| 13 | CatBoost (advanced boosting) | ✅ | `notebooks/08_catboost.ipynb` |
| 14 | Stacking ensemble | ✅ | `notebooks/09_stacking.ipynb` |
| 15 | Hyperparameter tuning | ✅ | `src/tuning.py`, `reports/tuning/` |
| 16 | Cross-validation (StratifiedKFold 5, f1_macro) | ✅ | `reports/tuning/cv_fold_scores.csv` |
| 17 | Model evaluation (incl. OvR ROC-AUC) | ✅ | `reports/model_comparison.csv`, `reports/evaluation/` |
| 18 | Confusion matrices | ✅ | `figures/13_confusion_matrix_*.png`, `figures/14_confusion_matrix_final_model.png` |
| 19 | Per-class F1 analysis | ✅ | `reports/evaluation/per_class_metrics_all_models.csv` |
| 20 | Misclassification analysis | ✅ | `reports/misclassification_analysis.csv`, `reports/evaluation/class_boundary_evidence.csv` |
| 21 | Feature importance / SHAP | ✅ | `reports/feature_importance.csv`, `figures/feature_importance.png`, `figures/16_shap_*.png` |
| 22 | Model comparison | ✅ | `reports/model_comparison.csv`, `figures/15_model_comparison_macro_f1.png` |
| 23 | Final model selection (highest CV Macro F1 → XGBoost) | ✅ | `src/final_evaluation.py`, report ch. 22 |
| 24 | Model saved with Joblib (reload verified identical) | ✅ | `models/final_model.joblib`, `models/preprocessing_pipeline.joblib` |
| 25 | Streamlit deployment | ✅ | `app.py`, `src/inference.py` |
| 26 | Deployment tested | ✅ | `tests/test_app.py` (headless app runs), `figures/app_screenshots/` |
| 27 | Unit tests (36, all passing) | ✅ | `tests/` → `pytest` |
| 28 | Final report | ✅ | `reports/Final_Report.docx` (41 pages, 25 figures, 21 tables) |
| 29 | Presentation (18 slides) | ✅ | `reports/Presentation.pptx` |
| 30 | README | ✅ | `README.md` |
| 31 | requirements.txt | ✅ | `requirements.txt` |
| 32 | GitHub-ready structure | ✅ | `.gitignore`; the virtual environment is excluded |

## Deviations from the brief, and why

- **Target definition.** RNF is not a class, and 50 rows with contradictory, unknown or multiple failure
  labels were excluded. Every exclusion is logged with its reason. This was approved at Stage 2.
- **Final model chosen on CV, not test.** Stacking has a higher *test* Macro F1 (0.802 vs 0.775). Selecting
  it on that basis would use the test set for model selection, so the pre-defined CV rule was kept. The
  trade-off (Stacking raises 105 false alarms, XGBoost 9) is reported in full.
- **Project structure.** A few modules were added to or renamed from the suggested layout (`feature_study.py`,
  `final_evaluation.py`, `inference.py`, `plotting.py`, `run_pipeline.py`), as the brief allowed. The final
  model is saved as one complete pipeline, with the preprocessing steps also saved separately.

## What remains (optional)

- Add a repository URL to the README's `git clone` line once the project is pushed to GitHub.
- Fill in any institution, course or supervisor details on the report's title page.
- In Word, the table of contents fills in when the file is opened. If prompted, allow it to update fields.
