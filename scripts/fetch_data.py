"""Download the pKa datasets into data/raw.

Everything the project trains on comes from here, so provenance lives in one
place and nothing is described from memory. Both sources are MIT licensed.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

CZODROWSKILAB = "https://raw.githubusercontent.com/czodrowskilab/Machine-learning-meets-pKa/master/datasets"
SAMPL6 = "https://raw.githubusercontent.com/samplchallenges/SAMPL6/master/physical_properties/pKa"

FILES = [
    (f"{CZODROWSKILAB}/combined_training_datasets_unique.sdf", "czodrowskilab_train.sdf"),
    (f"{CZODROWSKILAB}/novartis_cleaned_mono_unique_notraindata.sdf", "novartis_test.sdf"),
    (f"{CZODROWSKILAB}/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf", "avlilumove_test.sdf"),
    (f"{SAMPL6}/experimental_data/pKa_experimental_values_stacked.csv", "sampl6_stacked.csv"),
    (f"{SAMPL6}/experimental_data/experimental_microstates.csv", "sampl6_experimental_microstates.csv"),
    (f"{SAMPL6}/molecule_ID_and_SMILES.csv", "sampl6_molecules.csv"),
]
FILES += [
    (f"{SAMPL6}/microstates/SM{i:02d}_microstates.csv", f"sampl6_microstates/SM{i:02d}.csv")
    for i in range(1, 25)
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()
    root = Path(args.out)

    for url, name in FILES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            print(f"  have   {name}")
            continue
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = response.read()
        except Exception as exc:
            print(f"  FAILED {name}: {exc}")
            continue
        target.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()[:16]
        print(f"  got    {name:<44} {len(payload):>10,} B  sha256:{digest}")

    print(f"\nwrote to {root.resolve()}")


if __name__ == "__main__":
    main()
