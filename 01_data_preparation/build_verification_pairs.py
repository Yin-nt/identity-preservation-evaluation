"""Build balanced identity-verification pairs.

Turns a per-identity index CSV into a balanced set of genuine and impostor
pairs for face verification:

  * genuine  (label=1): each candidate compared against its own reference.
  * impostor (label=0): each candidate compared against another identity's
                        reference (obtained by rolling the reference column by
                        one), yielding one impostor pair per identity.

The genuine and impostor rows are concatenated and shuffled.

Usage:
    python build_verification_pairs.py --input pairs.csv --output pairs_balanced.csv
"""

import argparse

import numpy as np
import pandas as pd


def create_balanced_csv(input_csv, output_csv):
    df_match = pd.read_csv(input_csv)
    df_match["label"] = 1
    df_match["person_ref"] = df_match["person"]  # same identity

    # Impostor pairs: shift ref_path/person by one so identity A is compared
    # against identity B's reference.
    df_mismatch = df_match.copy()
    df_mismatch["label"] = 0
    df_mismatch["ref_path"] = np.roll(df_mismatch["ref_path"], 1)
    df_mismatch["person_ref"] = np.roll(df_mismatch["person"], 1)

    df_final = pd.concat([df_match, df_mismatch], ignore_index=True)
    df_final = df_final.sample(frac=1).reset_index(drop=True)
    df_final.to_csv(output_csv, index=False)

    print(f"Saved balanced dataset: {output_csv}")
    print(
        f"{len(df_match)} genuine + {len(df_mismatch)} impostor "
        f"= {len(df_final)} pairs."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build balanced verification pairs.")
    parser.add_argument("--input", required=True, help="Per-identity index CSV.")
    parser.add_argument("--output", required=True, help="Output balanced-pairs CSV.")
    args = parser.parse_args()

    create_balanced_csv(args.input, args.output)
