import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
DIPEPTIDES = [a + b for a in AMINO_ACIDS for b in AMINO_ACIDS]


def encode(seq):
    n = len(seq)
    aa = np.array([seq.count(a) / n for a in AMINO_ACIDS])
    dp = np.array([seq.count(d) / (n - 1) for d in DIPEPTIDES])
    nt = np.array([1 if seq[0] == a else 0 for a in AMINO_ACIDS])
    ct = np.array([1 if seq[-1] == a else 0 for a in AMINO_ACIDS])
    return np.concatenate([aa, dp, [n], nt, ct])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = joblib.load(args.model)
    mask = np.load(args.mask)
    df = pd.read_csv(args.input)
    seqs = df["sequence"].astype(str).str.strip().str.upper().tolist()
    x = np.array([encode(s) for s in seqs])[:, mask]
    prob = model.predict_proba(x)[:, 1]
    out = pd.DataFrame({"sequence": seqs, "probability": prob})
    out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
