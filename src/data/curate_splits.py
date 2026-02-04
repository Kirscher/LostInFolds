

import argparse
import json
from pathlib import Path
from typing import Dict, List

from sklearn.model_selection import KFold


def main_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_splits_path", type=str, help="Path to the original splits JSON file.", required=True)
    parser.add_argument("--output_splits_path", type=str, help="Path to save the new splits JSON file.", required=True)
    parser.add_argument("--rater_split", type=str, help="Suffix used to denote rater split position in case IDs.", required=True)
    parser.add_argument("--num_splits", type=int, default=5, help="Number of splits to create.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    args.original_splits_path = Path(args.original_splits_path)
    args.output_splits_path = Path(args.output_splits_path)
    return args   


def check_split_integrity(splits: List[Dict[str, List[str]]], expected_total_cases: int, rater_split: str) -> None:
    for i in range(len(splits)):
        train_split_incl_rater = splits[i]["train"]
        val_split_incl_rater = splits[i]["val"]

        train_split_cases = set([x.split(rater_split)[0] for x in train_split_incl_rater])
        val_split_cases = set([x.split(rater_split)[0] for x in val_split_incl_rater])
        assert len(train_split_cases.intersection(val_split_cases)) == 0, "Train and Val splits have overlapping cases!"
        assert len(train_split_incl_rater) + len(val_split_incl_rater) == expected_total_cases, "Some cases are missing in the splits!"
        print(f"Split {i}: Train Cases: {len(train_split_incl_rater)}, Val Cases: {len(val_split_incl_rater)}")


def create_splits(original_splits_path: Path, output_splits_path: Path, num_splits: int, random_seed: int, rater_split: str) -> None:
    with open(original_splits_path, "r") as f:
        splits = json.load(f)
    print()
    train_0 = splits[0]["train"]
    val_0 = splits[0]["val"]
    cases_incl_rater = train_0 + val_0

    train_0_cases = set([x.split(rater_split)[0] for x in train_0])
    val_0_cases = set([x.split(rater_split)[0] for x in val_0])
    cases = sorted(list(train_0_cases.union(val_0_cases)))
    print(f"Total Cases: {len(cases)}")
    print(f"Total Cases incl Rater: {len(cases_incl_rater)}")

    kf = KFold(n_splits=num_splits, shuffle=True, random_state=random_seed)
    splits = []
    for train_index, val_index in kf.split(cases):
        train_cases = [cases[i] for i in train_index]
        val_cases = [cases[i] for i in val_index]
        train_cases_rater = sorted([case_rater for case_rater in cases_incl_rater if case_rater.split(rater_split)[0] in train_cases])
        val_cases_rater = sorted([case_rater for case_rater in cases_incl_rater if case_rater.split(rater_split)[0] in val_cases])
        splits.append({"train": train_cases_rater, "val": val_cases_rater})
    
    check_split_integrity(splits, expected_total_cases=len(cases_incl_rater), rater_split=rater_split)
    with open(output_splits_path, "w") as f:
        json.dump(splits, f, indent=4)  


if __name__=="__main__":
    args = main_cli()
    create_splits(
        original_splits_path=args.original_splits_path,
        output_splits_path=args.output_splits_path,
        num_splits=args.num_splits,
        random_seed=args.random_seed,
        rater_split=args.rater_split
    )