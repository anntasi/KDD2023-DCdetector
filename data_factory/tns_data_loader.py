# data_factory/tns_data_loader.py
# English comment: Dataset and dataloader for TNS PDU-level features grouped by conn_id.

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def _load_sessions_index_csv(path: Path) -> Dict[str, List[int]]:
    """
    English comment: Load sessions_index.csv that maps conn_id -> list of row indices.
    Supports two common formats:
      1) columns: conn_id, indices (JSON list)
      2) columns: conn_id, row_indices (JSON list)
    """
    df = pd.read_csv(path)
    # Try best-effort column detection
    if "conn_id" not in df.columns:
        raise ValueError("sessions_index.csv must contain column 'conn_id'")

    cand_cols = [c for c in ["indices", "row_indices", "rows"] if c in df.columns]
    if not cand_cols:
        raise ValueError("sessions_index.csv must contain an indices column (e.g., 'indices') with JSON list")

    idx_col = cand_cols[0]
    out: Dict[str, List[int]] = {}
    for _, r in df.iterrows():
        cid = str(r["conn_id"])
        s = r[idx_col]
        # JSON list string -> python list
        if isinstance(s, str):
            arr = json.loads(s)
        else:
            arr = list(s)
        out[cid] = [int(x) for x in arr]
    return out


class TNSWindowDataset(Dataset):
    """
    English comment:
    Build sliding windows within each conn_id. Never cross session boundary.
    Each item: window tensor of shape [win_size, D]
    Also keep meta mapping so we can later assign scores back to PDU rows.
    """

    def __init__(
        self,
        pdu_features: np.ndarray,
        sessions_map: Dict[str, List[int]],
        win_size: int,
        stride: int = 1,
        min_session_len: int | None = None,
    ):
        self.win_size = int(win_size)
        self.stride = int(stride)
        self.min_session_len = int(min_session_len) if min_session_len is not None else self.win_size

        self.samples: List[Tuple[np.ndarray, List[int], str]] = []
        # sample = (window_feats, window_row_indices, conn_id)

        for cid, rows in sessions_map.items():
            rows = list(rows)
            if len(rows) < self.min_session_len:
                continue

            feats = pdu_features[rows]  # (T, D)
            T = feats.shape[0]
            for start in range(0, T - self.win_size + 1, self.stride):
                end = start + self.win_size
                w = feats[start:end]
                w_rows = rows[start:end]
                self.samples.append((w, w_rows, cid))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        w, w_rows, cid = self.samples[idx]
        x = torch.tensor(w, dtype=torch.float32)
        return x, w_rows, cid


def build_tns_loaders(
    data_root: str,
    win_size: int,
    batch_size: int,
    num_workers: int = 0,
    stride: int = 1,
):
    """
    English comment:
    For unsupervised setting with no labels, we will:
      - Use all windows for training
      - Reuse same loader for testing/inference
    """
    root = Path(data_root)
    pdu_features = np.load(root / "pdu_features.npy")  # (N, D)
    sessions_map = _load_sessions_index_csv(root / "sessions_index.csv")

    # Basic sanity check
    if pdu_features.ndim != 2:
        raise ValueError(f"pdu_features.npy must be 2D, got shape {pdu_features.shape}")

    ds = TNSWindowDataset(
        pdu_features=pdu_features,
        sessions_map=sessions_map,
        win_size=win_size,
        stride=stride,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    # We return (train_loader, valid_loader, test_loader) to match common repo patterns.
    return loader, loader, loader
