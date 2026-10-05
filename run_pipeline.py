"""Run the whole training pipeline, stage by stage, from the raw CSV.

    python run_pipeline.py            # all stages
    python run_pipeline.py --from tuning

Stages run in the order below. Each one reads only the artefacts produced by
the stages before it, so the test split is first read in 'final_evaluation'.
"""

import argparse
import importlib
import time

STAGES = [
    ("data_understanding", "Stage 2  - dataset understanding"),
    ("preprocessing", "Stage 3  - cleaning & preprocessing"),
    ("eda", "Stage 4  - exploratory data analysis"),
    ("feature_study", "Stages 5-6 - feature engineering/selection, train-test split"),
    ("tuning", "Stages 7-12 - baseline, RF, XGBoost, CatBoost, stacking, tuning, CV"),
    ("final_evaluation", "Stages 13-17 - test evaluation, comparison, interpretation, save"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", choices=[name for name, _ in STAGES], default=STAGES[0][0],
                        help="first stage to run (earlier stages' outputs must already exist)")
    args = parser.parse_args()

    names = [name for name, _ in STAGES]
    for name, title in STAGES[names.index(args.start):]:
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)
        start = time.time()
        importlib.import_module(f"src.{name}").main()
        print(f"[{name} finished in {time.time() - start:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
