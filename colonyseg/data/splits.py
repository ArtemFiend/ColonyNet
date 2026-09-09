"""Split by source image so offline augmentations cannot cross partitions."""
import random
import re


def source_id(sample_id: str) -> str:
    # Matches the project's augmentation tools, including repeated augmentation.
    return re.sub(r"(?:__soft\d+)+$", "", sample_id)


def validate_split(train_ids: list[str], val_ids: list[str]) -> None:
    if not train_ids or not val_ids:
        raise ValueError("Training and validation partitions must both contain samples.")
    overlap = {source_id(i) for i in train_ids} & {source_id(i) for i in val_ids}
    if overlap:
        raise ValueError(f"Dataset leakage: {len(overlap)} source image(s) occur in both partitions.")


def split_ids(all_ids: list[str], val_fraction: float, seed: int = 42):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be strictly between 0 and 1.")
    groups = sorted({source_id(i) for i in all_ids})
    if len(groups) < 2:
        raise ValueError("At least two independent source images are required.")
    random.Random(seed).shuffle(groups)
    n_val = min(len(groups) - 1, max(1, int(len(groups) * val_fraction)))
    validation = set(groups[:n_val])
    train = [i for i in all_ids if source_id(i) not in validation]
    val = [i for i in all_ids if source_id(i) in validation]
    validate_split(train, val)
    return train, val
