import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from xgboost import XGBClassifier

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
DIPEPTIDES = [a + b for a in AMINO_ACIDS for b in AMINO_ACIDS]
FEATURE_NAMES = (
    [f"AA_{a}" for a in AMINO_ACIDS]
    + [f"DP_{d}" for d in DIPEPTIDES]
    + ["Seq_Length"]
    + [f"N_term_{a}" for a in AMINO_ACIDS]
    + [f"C_term_{a}" for a in AMINO_ACIDS]
)


def read_data(path):
    df = pd.read_csv(path)
    seqs = df["sequence"].astype(str).str.strip().str.upper().tolist()
    labels = df["label"].astype(int).to_numpy()
    return seqs, labels


def encode(seq):
    n = len(seq)
    aa = np.array([seq.count(a) / n for a in AMINO_ACIDS])
    dp = np.array([seq.count(d) / (n - 1) for d in DIPEPTIDES])
    nt = np.array([1 if seq[0] == a else 0 for a in AMINO_ACIDS])
    ct = np.array([1 if seq[-1] == a else 0 for a in AMINO_ACIDS])
    return np.concatenate([aa, dp, [n], nt, ct])


def build_matrix(seqs):
    return np.array([encode(s) for s in seqs])


def select_features(x, y, seed, k):
    model = LGBMClassifier(
        n_estimators=200,
        random_state=seed,
        device="cpu",
        n_jobs=-1,
        is_unbalance=True,
    )
    model.fit(x, y)
    gain = model.booster_.feature_importance(importance_type="gain")
    return np.argsort(gain)[::-1][:k]


def make_model(params, scale_pos_weight):
    lgbm_params = {
        k: v for k, v in params.items() if k not in ["path_smooth", "extra_trees", "device"]
    }
    lgbm_params.update({"random_state": 42, "verbose": -1, "n_jobs": -1})
    xgb_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "reg_alpha": 0.5,
        "reg_lambda": 0.5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    return VotingClassifier(
        estimators=[
            ("lgbm", LGBMClassifier(**lgbm_params)),
            ("xgb", XGBClassifier(**xgb_params)),
        ],
        voting="soft",
    )


def tune(x, y, seed, n_trials):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_est", 100, 500, step=50),
            "learning_rate": trial.suggest_float("lr", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child", 3, 20),
            "reg_alpha": trial.suggest_float("alpha", 0.1, 20.0, log=True),
            "reg_lambda": trial.suggest_float("lambda", 0.1, 10.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample", 0.5, 0.9),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "subsample_freq": 1,
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 15.0),
            "extra_trees": True,
            "path_smooth": trial.suggest_float("path_smooth", 0.0, 10.0),
            "random_state": 42,
            "verbose": -1,
            "device": "cpu",
        }
        scores = []
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
        for tr, te in cv.split(x, y):
            model = LGBMClassifier(**params)
            model.fit(x[tr], y[tr])
            prob = model.predict_proba(x[te])[:, 1]
            fold_scores = [
                matthews_corrcoef(y[te], (prob >= t).astype(int))
                for t in np.linspace(0.2, 0.8, 30)
            ]
            scores.append(max(fold_scores))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def cross_validate(x, y, params):
    model = make_model(params, params.get("scale_pos_weight", 1.0))
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr, te in kf.split(x, y):
        model.fit(x[tr], y[tr])
        oof[te] = model.predict_proba(x[te])[:, 1]
    thresholds = np.linspace(0.1, 0.9, 200)
    mcc = [matthews_corrcoef(y, (oof >= t).astype(int)) for t in thresholds]
    threshold = float(thresholds[int(np.argmax(mcc))])
    pred = (oof >= threshold).astype(int)
    return {
        "mcc": float(matthews_corrcoef(y, pred)),
        "auc": float(roc_auc_score(y, oof)),
        "threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--features", type=int, default=150)
    args = parser.parse_args()

    seqs, y = read_data(args.data)
    x_raw = build_matrix(seqs)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    best = None
    metrics = None
    for r in range(args.repeats):
        seed = 42 + r
        mask = select_features(x_raw, y, seed, args.features)
        x = x_raw[:, mask]
        params = tune(x, y, seed, args.trials)
        result = cross_validate(x, y, params)
        if best is None or result["mcc"] > metrics["mcc"]:
            best = (mask, params, result)
            metrics = result
            joblib.dump(make_model(params, params.get("scale_pos_weight", 1.0)), out / "ensemble.pkl")
            np.save(out / "feature_mask.npy", mask)
            joblib.dump(params, out / "lgbm_params.pkl")
            with open(out / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
