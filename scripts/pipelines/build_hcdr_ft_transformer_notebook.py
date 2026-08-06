#!/usr/bin/env python3
"""Build the focused HCDR Stage C FT-Transformer Kaggle notebook."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = ROOT / "notebooks/kaggle/hcdr-stage-c-model-benchmarks"
NOTEBOOK_PATH = NOTEBOOK_DIR / "hcdr-ft-transformer-stage-c.ipynb"


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source.strip() + "\n")


def markdown(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source.strip() + "\n")


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        markdown(
            """
# Home Credit Default Risk — FT-Transformer

## Why

The Stage C benchmark is dominated by tree boosting. This experiment asks one
focused question: can an FT-Transformer learn useful interactions across the
159 numeric and 16 categorical Stage C features while preserving the existing
leakage-safe split contract?

Training is healthy when the first optimizer step has finite loss, logits, and
gradients, and validation AUC can be computed. Final success is test AUC/Gini/KS
after model selection by validation AUC only.

## How

- Each numeric value is converted into its own learnable feature token.
- Each categorical value uses a train-only vocabulary and embedding table.
- A learned `[CLS]` token and Transformer encoder model cross-feature attention.
- Weighted binary cross-entropy addresses the 8.07% event-rate imbalance.
- Early stopping reads validation AUC only; the test split remains untouched
  until the best checkpoint is selected.

The 60/20/20 split is stratified random, not out-of-time validation. This is a
research benchmark, not a production credit-approval model.
"""
        ),
        code(
            """
from __future__ import annotations

import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

RANDOM_STATE = 42
SMOKE_TEST = bool(int(os.getenv("HCDR_SMOKE_TEST", "0")))
OUTPUT_DIR = Path(os.getenv("HCDR_OUTPUT_DIR", "/kaggle/working" if Path("/kaggle/working").exists() else "/tmp/hcdr-ft-transformer-smoke"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
GPU_CAPABILITY = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
GPU_ARCH = (
    f"sm_{GPU_CAPABILITY[0]}{GPU_CAPABILITY[1]}"
    if GPU_CAPABILITY is not None
    else None
)
SUPPORTED_CUDA_ARCHES = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
CUDA_COMPATIBLE = GPU_ARCH is not None and GPU_ARCH in SUPPORTED_CUDA_ARCHES
if not SMOKE_TEST and not CUDA_COMPATIBLE:
    raise RuntimeError(
        "A CUDA-compatible GPU is required for the full run: "
        f"gpu={GPU_NAME!r}, arch={GPU_ARCH!r}, "
        f"torch_arches={SUPPORTED_CUDA_ARCHES!r}"
    )
DEVICE = torch.device("cuda" if CUDA_COMPATIBLE else "cpu")
BATCH_SIZE = 64 if SMOKE_TEST else 256
MAX_EPOCHS = 2 if SMOKE_TEST else 15
PATIENCE = 2 if SMOKE_TEST else 3
NUM_WORKERS = 0 if SMOKE_TEST else 2

print({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "sklearn": sklearn.__version__,
    "device": str(DEVICE),
    "gpu": GPU_NAME,
    "gpu_arch": GPU_ARCH,
    "torch_cuda_arches": SUPPORTED_CUDA_ARCHES,
    "cuda_compatible": CUDA_COMPATIBLE,
    "smoke_test": SMOKE_TEST,
})
"""
        ),
        code(
            """
KAGGLE_INPUT = Path("/kaggle/input/hcdr-stage-c-feature-matrix")
LOCAL_INPUT = Path("datasets/processed/hcdr")
LOCAL_RAW = Path("datasets/raw/home-credit-default-risk")

if (KAGGLE_INPUT / "feature_matrix.parquet").exists():
    input_dir = KAGGLE_INPUT
    sample_path = input_dir / "sample_submission.csv"
else:
    input_dir = LOCAL_INPUT
    sample_path = LOCAL_RAW / "sample_submission.csv"

matrix = pd.read_parquet(input_dir / "feature_matrix.parquet")
membership = pd.read_csv(input_dir / "split_membership.csv")
sample_submission = pd.read_csv(sample_path)

assert matrix.shape == (356255, 177), matrix.shape
assert matrix["SK_ID_CURR"].is_unique
assert membership["SK_ID_CURR"].is_unique
assert set(membership["split"]) == {"train", "valid", "test"}

labeled = matrix.loc[matrix["TARGET"].notna()].merge(
    membership, on="SK_ID_CURR", how="left", validate="one_to_one"
)
competition = matrix.loc[matrix["TARGET"].isna()].copy()
assert set(membership["SK_ID_CURR"]) == set(labeled["SK_ID_CURR"])
assert competition["SK_ID_CURR"].astype(int).reset_index(drop=True).equals(
    sample_submission["SK_ID_CURR"].astype(int).reset_index(drop=True)
)

if SMOKE_TEST:
    def stratified_sample(frame: pd.DataFrame, n: int) -> pd.DataFrame:
        if len(frame) <= n:
            return frame.copy()
        selected, _ = train_test_split(
            frame,
            train_size=n,
            stratify=frame["TARGET"],
            random_state=RANDOM_STATE,
        )
        return selected.copy()

    labeled = pd.concat(
        [
            stratified_sample(labeled.loc[labeled["split"].eq(split)], n)
            for split, n in (("train", 2048), ("valid", 1024), ("test", 1024))
        ],
        ignore_index=True,
    )
    competition = competition.head(1024).copy()
    sample_submission = sample_submission.head(1024).copy()

train = labeled.loc[labeled["split"].eq("train")].copy()
valid = labeled.loc[labeled["split"].eq("valid")].copy()
test = labeled.loc[labeled["split"].eq("test")].copy()
feature_columns = [column for column in matrix.columns if column not in {"SK_ID_CURR", "TARGET"}]
categorical_columns = list(train[feature_columns].select_dtypes(include=["object", "string", "category"]).columns)
numeric_columns = [column for column in feature_columns if column not in categorical_columns]

assert len(feature_columns) == 175
assert len(categorical_columns) == 16
assert len(numeric_columns) == 159
print({
    "input": str(input_dir),
    "splits": {name: len(frame) for name, frame in (("train", train), ("valid", valid), ("test", test), ("competition", competition))},
    "features": {"numeric": len(numeric_columns), "categorical": len(categorical_columns)},
    "bad_rate": float(train["TARGET"].mean()),
})
"""
        ),
        code(
            """
numeric_median = train[numeric_columns].median()
train_numeric = train[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(numeric_median)
numeric_mean = train_numeric.mean()
numeric_std = train_numeric.std().replace(0.0, 1.0).fillna(1.0)


def transform_numeric(frame: pd.DataFrame) -> np.ndarray:
    values = frame[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(numeric_median)
    return ((values - numeric_mean) / numeric_std).to_numpy(dtype="float32")


category_vocabularies: dict[str, dict[str, int]] = {}
category_cardinalities: list[int] = []
for column in categorical_columns:
    values = train[column].astype("string").fillna("__MISSING__")
    vocabulary = {value: index + 1 for index, value in enumerate(sorted(values.unique()))}
    category_vocabularies[column] = vocabulary
    category_cardinalities.append(len(vocabulary) + 1)  # zero is unknown


def transform_categories(frame: pd.DataFrame) -> np.ndarray:
    encoded = np.zeros((len(frame), len(categorical_columns)), dtype="int64")
    for index, column in enumerate(categorical_columns):
        values = frame[column].astype("string").fillna("__MISSING__")
        encoded[:, index] = values.map(category_vocabularies[column]).fillna(0).to_numpy(dtype="int64")
    return encoded


class CreditDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, *, with_target: bool = True) -> None:
        self.numeric = torch.from_numpy(transform_numeric(frame))
        self.categorical = torch.from_numpy(transform_categories(frame))
        self.target = (
            torch.from_numpy(frame["TARGET"].to_numpy(dtype="float32"))
            if with_target
            else None
        )

    def __len__(self) -> int:
        return len(self.numeric)

    def __getitem__(self, index: int):
        if self.target is None:
            return self.numeric[index], self.categorical[index]
        return self.numeric[index], self.categorical[index], self.target[index]


train_dataset = CreditDataset(train)
valid_dataset = CreditDataset(valid)
test_dataset = CreditDataset(test)
competition_dataset = CreditDataset(competition, with_target=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda")
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda")
competition_loader = DataLoader(competition_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda")

print("category cardinalities", category_cardinalities)
"""
        ),
        code(
            """
class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, feature_count: int, token_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(feature_count, token_dim))
        self.bias = nn.Parameter(torch.empty(feature_count, token_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class CategoricalFeatureTokenizer(nn.Module):
    def __init__(self, cardinalities: list[int], token_dim: int) -> None:
        super().__init__()
        offsets = np.cumsum([0, *cardinalities[:-1]])
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.embedding = nn.Embedding(sum(cardinalities), token_dim)
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.embedding(values + self.offsets.unsqueeze(0))


class FTTransformer(nn.Module):
    def __init__(
        self,
        numeric_count: int,
        category_cardinalities: list[int],
        *,
        token_dim: int = 64,
        layers: int = 3,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.numeric_tokenizer = NumericalFeatureTokenizer(numeric_count, token_dim)
        self.categorical_tokenizer = CategoricalFeatureTokenizer(category_cardinalities, token_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=token_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(token_dim), nn.ReLU(), nn.Linear(token_dim, 1))

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat(
            [self.numeric_tokenizer(numeric), self.categorical_tokenizer(categorical)],
            dim=1,
        )
        cls = self.cls_token.expand(len(numeric), -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(encoded[:, 0]).squeeze(1)


MODEL_CONFIG = {
    "token_dim": 64,
    "layers": 3,
    "heads": 8,
    "dropout": 0.1,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
}
model = FTTransformer(
    len(numeric_columns),
    category_cardinalities,
    token_dim=MODEL_CONFIG["token_dim"],
    layers=MODEL_CONFIG["layers"],
    heads=MODEL_CONFIG["heads"],
    dropout=MODEL_CONFIG["dropout"],
).to(DEVICE)

parameter_count = sum(parameter.numel() for parameter in model.parameters())
print(model)
print("trainable parameters", parameter_count)
"""
        ),
        code(
            """
def predict(loader: DataLoader) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            numeric, categorical = batch[:2]
            logits = model(numeric.to(DEVICE, non_blocking=True), categorical.to(DEVICE, non_blocking=True))
            predictions.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(predictions)


event_count = float(train["TARGET"].sum())
non_event_count = float(len(train) - event_count)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(non_event_count / event_count, device=DEVICE))
optimizer = torch.optim.AdamW(model.parameters(), lr=MODEL_CONFIG["learning_rate"], weight_decay=MODEL_CONFIG["weight_decay"])
checkpoint_path = OUTPUT_DIR / "ft_transformer_best.pt"
history: list[dict[str, float | int]] = []
best_valid_auc = -np.inf
stale_epochs = 0
healthcheck_written = False

for epoch in range(1, MAX_EPOCHS + 1):
    started = time.perf_counter()
    model.train()
    loss_sum = 0.0
    example_count = 0
    for numeric, categorical, target in train_loader:
        numeric = numeric.to(DEVICE, non_blocking=True)
        categorical = categorical.to(DEVICE, non_blocking=True)
        target = target.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(numeric, categorical)
        loss = criterion(logits, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss: {loss.item()}")
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        if not gradients_finite:
            raise FloatingPointError("non-finite gradient detected")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if not healthcheck_written:
            healthcheck = {
                "status": "TRAINING_HEALTHCHECK_OK",
                "epoch": epoch,
                "batch_size": int(len(target)),
                "loss": float(loss.item()),
                "logit_min": float(logits.min().item()),
                "logit_max": float(logits.max().item()),
                "gradients_finite": gradients_finite,
                "device": str(DEVICE),
            }
            (OUTPUT_DIR / "training_healthcheck.json").write_text(json.dumps(healthcheck, indent=2) + "\\n", encoding="utf-8")
            print("TRAINING_HEALTHCHECK_OK", healthcheck, flush=True)
            healthcheck_written = True

        loss_sum += float(loss.item()) * len(target)
        example_count += len(target)

    valid_predictions = predict(valid_loader)
    valid_auc = float(roc_auc_score(valid["TARGET"], valid_predictions))
    row = {
        "epoch": epoch,
        "train_loss": loss_sum / example_count,
        "valid_auc": valid_auc,
        "seconds": time.perf_counter() - started,
    }
    history.append(row)
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    print("EPOCH", row, flush=True)

    if valid_auc > best_valid_auc + 1e-5:
        best_valid_auc = valid_auc
        stale_epochs = 0
        torch.save({"model_state": model.state_dict(), "config": MODEL_CONFIG, "valid_auc": valid_auc}, checkpoint_path)
    else:
        stale_epochs += 1
        if stale_epochs >= PATIENCE:
            print("EARLY_STOPPING", {"epoch": epoch, "best_valid_auc": best_valid_auc}, flush=True)
            break

assert healthcheck_written
"""
        ),
        code(
            """
checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint["model_state"])
valid_predictions = predict(valid_loader)
test_predictions = predict(test_loader)
competition_predictions = predict(competition_loader)


def ks_statistic(y_true: pd.Series, scores: np.ndarray) -> float:
    false_positive, true_positive, _ = roc_curve(y_true, scores)
    return float(np.max(true_positive - false_positive))


metric_rows = []
for split, labels, scores in (
    ("valid", valid["TARGET"], valid_predictions),
    ("test", test["TARGET"], test_predictions),
):
    auc = float(roc_auc_score(labels, scores))
    metric_rows.append({
        "model": "ft_transformer",
        "split": split,
        "n": len(labels),
        "bad_rate": float(labels.mean()),
        "auc": auc,
        "gini": 2.0 * auc - 1.0,
        "ks": ks_statistic(labels, scores),
    })

metrics = pd.DataFrame(metric_rows)
metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
submission = sample_submission.copy()
submission["TARGET"] = competition_predictions
assert list(submission.columns) == ["SK_ID_CURR", "TARGET"]
assert submission["SK_ID_CURR"].is_unique
assert np.isfinite(submission["TARGET"]).all()
assert submission["TARGET"].between(0.0, 1.0).all()
submission.to_csv(OUTPUT_DIR / "submission_ft_transformer.csv", index=False)

experiment = {
    "model": "FT-Transformer",
    "model_config": MODEL_CONFIG,
    "parameter_count": parameter_count,
    "random_state": RANDOM_STATE,
    "split": "precomputed stratified random 60/20/20; not out-of-time",
    "feature_count": len(feature_columns),
    "numeric_count": len(numeric_columns),
    "categorical_count": len(categorical_columns),
    "category_cardinalities": category_cardinalities,
    "best_valid_auc": best_valid_auc,
    "device": str(DEVICE),
    "torch_version": torch.__version__,
    "smoke_test": SMOKE_TEST,
}
(OUTPUT_DIR / "experiment_config.json").write_text(json.dumps(experiment, indent=2) + "\\n", encoding="utf-8")
display(metrics)
"""
        ),
        markdown(
            """
## Interpretation limits

- Validation/test results come from one fixed random split and do not measure
  temporal drift.
- The class-weighted loss changes optimization, not the underlying event rate.
- The test split is evaluated only after validation selects the checkpoint.
- Offline performance does not establish calibration, fairness, compliance, or
  production suitability.
"""
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    return notebook


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook(), NOTEBOOK_PATH)
    metadata = {
        "id": "cyclerlol/hcdr-ft-transformer-stage-c",
        "title": "HCDR FT Transformer Stage C",
        "code_file": NOTEBOOK_PATH.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["cyclerlol/hcdr-stage-c-feature-matrix"],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    (NOTEBOOK_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
