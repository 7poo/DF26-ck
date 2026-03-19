
import os
import gc
import math
import time
import copy
import random
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# v46 — full-train only, snapshot-ensemble, regime-aware hybrid
# Built from the strongest parts of v44:
# - shared sequence encoder with prefix/suffix/middle pooling
# - full-train only on train+val
# - exact-first postproc
# New additions:
# - dual-head for attr_3 / attr_6: regression + 100-class distribution
# - regime-aware residual experts for start / mid / end / attr6
# - snapshot ensemble from one training run (no separate retraining)
# - safe soft memory for attr_3
# ============================================================

# -------------------------------
# SYSTEM
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = torch.cuda.is_available()
PIN_MEMORY = torch.cuda.is_available()
DETERMINISTIC = os.getenv("DETERMINISTIC", "1") == "1"
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = not DETERMINISTIC
    torch.backends.cudnn.allow_tf32 = not DETERMINISTIC
    torch.backends.cudnn.benchmark = not DETERMINISTIC
    torch.backends.cudnn.deterministic = DETERMINISTIC
    if not DETERMINISTIC:
        torch.set_float32_matmul_precision("high")
else:
    torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {props.total_memory/1e9:.1f} GB")


# -------------------------------
# CONFIG
# -------------------------------
SEED = 42
PAD = 0
UNK = 1
SEP_TOKEN = 103
TOKEN_104 = 104
MAX_SESSIONS = 6
HASH2 = 4096
HASH3 = 8192
LEN_BUCKETS = 67

SCRIPT_DIR = Path(__file__).resolve().parent

def resolve_data_dir() -> str:
    env_data_dir = os.getenv("DATA_DIR")
    candidates = []
    if env_data_dir:
        candidates.append(Path(env_data_dir).expanduser())
    candidates.extend(
        [
            SCRIPT_DIR,
            Path.cwd(),
            Path("/content/sample_data"),
            Path("sample_data"),
            Path("/mnt/data"),
        ]
    )
    required = ["X_train.csv", "X_val.csv", "X_test.csv", "Y_train.csv", "Y_val.csv"]
    for candidate in candidates:
        candidate = candidate.resolve()
        if all((candidate / file_name).exists() for file_name in required):
            return str(candidate)
    raise FileNotFoundError(
        "Could not locate dataset directory. Set DATA_DIR or place the CSV files next to Model.py."
    )

DATA_DIR = resolve_data_dir()
TRAIN_X = os.path.join(DATA_DIR, "X_train.csv")
VAL_X   = os.path.join(DATA_DIR, "X_val.csv")
TEST_X  = os.path.join(DATA_DIR, "X_test.csv")
TRAIN_Y = os.path.join(DATA_DIR, "Y_train.csv")
VAL_Y   = os.path.join(DATA_DIR, "Y_val.csv")

TARGET_COLS = [f"attr_{i}" for i in range(1, 7)]
FEATURE_COLS = [f"feature_{i}" for i in range(1, 67)]

USE_COMPILE = False
FINAL_EPOCHS = 120              # 96 if bạn muốn an toàn thời gian hơn
TRAIN_BS = 4096
PRED_BS = 16384
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
LR = 1.6e-3
WEIGHT_DECAY = 1e-4
WARMUP_PCT = 0.08
SNAPSHOT_EPOCHS = None          # auto: [70%, 85%, 100%] of FINAL_EPOCHS

D_MODEL = 224
N_LAYERS = 4
N_HEADS = 8
FF_DIM = 896
DROPOUT = 0.10
SEQ_LEN = 66

AUX_EV_WEIGHT = 0.20            # keep ordinal support light; CE stays primary for month/day
AUX_CE_WEIGHT = 1.0
# attr_3/6 use curriculum weights inside model_loss() instead of extreme fixed weights

SUB_NAME = "submission_v46_snapshot_regime_dualhead.csv"
CKPT_NAME = "seqmodel_v46_snapshot_regime_dualhead.ckpt"


# -------------------------------
# REPRO
# -------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if DETERMINISTIC:
        torch.use_deterministic_algorithms(True, warn_only=True)


def stable_hash(values, mod):
    if len(values) == 0:
        return 0
    h = 1469598103934665603
    for v in values:
        h ^= int(v) + 0x9E3779B97F4A7C15
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return 1 + (h % (mod - 1))


def strip104(seq):
    if len(seq) > 0 and seq[-1] == TOKEN_104:
        return seq[:-1]
    return seq


def bucket_len(n):
    return int(max(0, min(n, LEN_BUCKETS - 1)))


def decode_preds(pred_enc):
    pred = pred_enc.copy().astype(np.int64)
    pred[:, 0] += 1
    pred[:, 1] += 1
    pred[:, 3] += 1
    pred[:, 4] += 1
    pred[:, 2] = np.clip(pred[:, 2], 0, 99)
    pred[:, 5] = np.clip(pred[:, 5], 0, 99)
    return pred


# -------------------------------
# DATA
# -------------------------------
def load_csvs():
    x_tr = pd.read_csv(TRAIN_X)
    x_va = pd.read_csv(VAL_X)
    x_te = pd.read_csv(TEST_X)
    y_tr = pd.read_csv(TRAIN_Y)
    y_va = pd.read_csv(VAL_Y)
    x_full = pd.concat([x_tr, x_va], axis=0, ignore_index=True)
    y_full = pd.concat([y_tr, y_va], axis=0, ignore_index=True)
    return x_full, y_full, x_te


def build_vocab(x_full, x_test):
    uniq = set()
    for df in [x_full, x_test]:
        raw_np = df[FEATURE_COLS].to_numpy(dtype=np.float32)
        for i in range(len(df)):
            row = [int(v) for v in raw_np[i] if int(v) != 0]
            uniq.update(row)
    token2id = {tok: i + 2 for i, tok in enumerate(sorted(uniq))}
    id2token = {v: k for k, v in token2id.items()}
    return token2id, id2token


def encode_targets(y_full):
    y = y_full[TARGET_COLS].copy()
    enc = np.zeros((len(y), 6), dtype=np.int64)
    enc[:, 0] = y["attr_1"].values - 1
    enc[:, 1] = y["attr_2"].values - 1
    enc[:, 2] = y["attr_3"].values
    enc[:, 3] = y["attr_4"].values - 1
    enc[:, 4] = y["attr_5"].values - 1
    enc[:, 5] = y["attr_6"].values
    raw = y.values.astype(np.int64)
    return enc, raw


def encode_rows(df, token2id):
    n = len(df)
    raw_np = df[FEATURE_COLS].to_numpy(dtype=np.float32)

    tokens = np.zeros((n, SEQ_LEN), dtype=np.int64)
    sessions = np.zeros((n, SEQ_LEN), dtype=np.int64)
    splits = np.zeros((n, SEQ_LEN), dtype=np.int64)
    positions = np.tile(np.arange(SEQ_LEN, dtype=np.int64), (n, 1))
    rev_positions = np.zeros((n, SEQ_LEN), dtype=np.int64)
    mask = np.zeros((n, SEQ_LEN), dtype=np.bool_)

    regime = np.zeros(n, dtype=np.int64)
    has103 = np.zeros(n, dtype=np.int64)
    has104 = np.zeros(n, dtype=np.int64)
    has609 = np.zeros(n, dtype=np.int64)
    starts609 = np.zeros(n, dtype=np.int64)
    end104 = np.zeros(n, dtype=np.int64)
    seq_len_arr = np.zeros(n, dtype=np.int64)
    len_bucket = np.zeros(n, dtype=np.int64)
    num103 = np.zeros(n, dtype=np.int64)
    last_seg_len = np.zeros(n, dtype=np.int64)

    f2 = np.zeros(n, dtype=np.int64)
    f3 = np.zeros(n, dtype=np.int64)
    f4 = np.zeros(n, dtype=np.int64)
    l2 = np.zeros(n, dtype=np.int64)
    l3 = np.zeros(n, dtype=np.int64)
    m2 = np.zeros(n, dtype=np.int64)
    rm2 = np.zeros(n, dtype=np.int64)

    raw_rows = []
    ids = df["id"].astype(str).tolist()

    for i in range(n):
        row = [int(v) for v in raw_np[i] if int(v) != 0][:SEQ_LEN]
        raw_rows.append(row)
        L = len(row)
        seq_len_arr[i] = L
        len_bucket[i] = bucket_len(L)
        if L == 0:
            continue

        arr = np.array([token2id.get(t, UNK) for t in row], dtype=np.int64)
        tokens[i, :L] = arr
        mask[i, :L] = True
        rev_positions[i, :L] = np.arange(L - 1, -1, -1, dtype=np.int64)

        has103[i] = int(SEP_TOKEN in row)
        has104[i] = int(TOKEN_104 in row)
        has609[i] = int(609 in row)
        starts609[i] = int(row[0] == 609)
        end104[i] = int(row[-1] == TOKEN_104)

        pos103 = [j for j, t in enumerate(row) if t == SEP_TOKEN]
        num103[i] = len(pos103)
        last_103 = pos103[-1] if pos103 else -1
        if last_103 >= 0:
            regime[i] = 0
        elif TOKEN_104 not in row:
            regime[i] = 1
        else:
            regime[i] = 2

        after = row[last_103 + 1:] if last_103 >= 0 else row
        after_strip = strip104(after)
        full_strip = strip104(row)
        last_seg_len[i] = len(after_strip)

        if len(row) >= 2:
            f2[i] = stable_hash(row[:2], HASH2)
            f4[i] = stable_hash(row[:min(4, len(row))], HASH3)
        if len(row) >= 3:
            f3[i] = stable_hash(row[:3], HASH3)
        if len(full_strip) >= 2:
            l2[i] = stable_hash(full_strip[-2:], HASH2)
        if len(full_strip) >= 3:
            l3[i] = stable_hash(full_strip[-3:], HASH3)
        if len(row) >= 6:
            m2[i] = stable_hash([row[4], row[5]], HASH2)
        if len(row) >= 8:
            rm2[i] = stable_hash([row[-6], row[-7]], HASH2)

        sess = 0
        split_flag = 0
        for pos, tok in enumerate(row):
            sessions[i, pos] = min(sess, MAX_SESSIONS - 1)
            splits[i, pos] = split_flag
            if tok == SEP_TOKEN:
                sess += 1
                split_flag = 1

    return {
        "tokens": tokens,
        "sessions": sessions,
        "splits": splits,
        "positions": positions,
        "rev_positions": rev_positions,
        "mask": mask,
        "regime": regime,
        "has103": has103,
        "has104": has104,
        "has609": has609,
        "starts609": starts609,
        "end104": end104,
        "seq_len": seq_len_arr,
        "len_bucket": len_bucket,
        "num103": num103,
        "last_seg_len": last_seg_len,
        "f2": f2,
        "f3": f3,
        "f4": f4,
        "l2": l2,
        "l3": l3,
        "m2": m2,
        "rm2": rm2,
        "raw_rows": raw_rows,
        "ids": ids,
    }


# -------------------------------
# LOOKUPS / POSTPROC
# -------------------------------
def make_det(buckets, min_n=5, min_pur=0.98):
    out = {}
    for k, vals in buckets.items():
        n = len(vals)
        if n < min_n:
            continue
        cnt = Counter(vals)
        best, c = cnt.most_common(1)[0]
        purity = c / n
        if purity >= min_pur:
            out[k] = best
    return out


def build_k_lookup(raw_rows, y_raw, k, front, attr_idx, strip=False, min_n=5, min_pur=0.98, regime_only=None):
    buckets = defaultdict(list)
    for i, row in enumerate(raw_rows):
        pos103 = [j for j, t in enumerate(row) if t == SEP_TOKEN]
        last_103 = pos103[-1] if pos103 else -1
        reg = 0 if last_103 >= 0 else (1 if TOKEN_104 not in row else 2)
        if regime_only is not None and reg != regime_only:
            continue
        seq = strip104(row) if strip else row
        if len(seq) < k:
            continue
        key = tuple(seq[:k]) if front else tuple(seq[-k:])
        buckets[key].append(int(y_raw[i, attr_idx]))
    return make_det(buckets, min_n=min_n, min_pur=min_pur)


def build_k_lookup_pair(raw_rows, y_raw, k, front, attr_indices, strip=False, min_n=10, min_pur=0.95):
    buckets = defaultdict(list)
    for i, row in enumerate(raw_rows):
        seq = strip104(row) if strip else row
        if len(seq) < k:
            continue
        key = tuple(seq[:k]) if front else tuple(seq[-k:])
        buckets[key].append(tuple(int(y_raw[i, j]) for j in attr_indices))
    return make_det(buckets, min_n=min_n, min_pur=min_pur)


def build_pospair_lookup(raw_rows, y_raw, positions, attr_idx, rev=False, min_n=5, min_pur=1.0):
    buckets = defaultdict(list)
    for i, row in enumerate(raw_rows):
        L = len(row)
        idxs = []
        ok = True
        for p in positions:
            idx = L - p if rev else p
            if idx < 0 or idx >= L:
                ok = False
                break
            idxs.append(idx)
        if not ok:
            continue
        key = tuple(row[j] for j in idxs)
        buckets[key].append(int(y_raw[i, attr_idx]))
    return make_det(buckets, min_n=min_n, min_pur=min_pur)


def build_pospair_stats(raw_rows, y_raw, positions, attr_idx, rev=False, min_n=12, max_std=4.0):
    out = {}
    buckets = defaultdict(list)
    for i, row in enumerate(raw_rows):
        L = len(row)
        idxs = []
        ok = True
        for p in positions:
            idx = L - p if rev else p
            if idx < 0 or idx >= L:
                ok = False
                break
            idxs.append(idx)
        if not ok:
            continue
        key = tuple(row[j] for j in idxs)
        buckets[key].append(int(y_raw[i, attr_idx]))
    for k, vals in buckets.items():
        if len(vals) < min_n:
            continue
        vals = np.asarray(vals, dtype=np.float32)
        std = float(vals.std())
        if std <= max_std:
            out[k] = (float(vals.mean()), len(vals), std)
    return out


def build_fullpattern(raw_rows, y_raw):
    before = defaultdict(list)
    after = defaultdict(list)
    full = defaultdict(list)
    for i, row in enumerate(raw_rows):
        full[tuple(row)].append(tuple(int(y_raw[i, j]) for j in range(6)))
        pos = [j for j, t in enumerate(row) if t == SEP_TOKEN]
        if not pos:
            continue
        p = pos[-1]
        before[tuple(row[:p])].append(tuple(int(y_raw[i, j]) for j in range(3)))
        after[tuple(row[p + 1:])].append(tuple(int(y_raw[i, j]) for j in range(3, 6)))
    return make_det(full, min_n=2, min_pur=1.0), make_det(before, min_n=2, min_pur=0.98), make_det(after, min_n=2, min_pur=0.98)


def build_lookups(raw_rows, y_raw):
    det_full, det_b, det_a = build_fullpattern(raw_rows, y_raw)
    return {
        "det_full": det_full,
        "det_b": det_b,
        "det_a": det_a,
        # start-date rules
        "f2a1": build_k_lookup(raw_rows, y_raw, 2, True, 0, strip=False, min_n=10, min_pur=0.98),
        "f3a1_exact": build_k_lookup(raw_rows, y_raw, 3, True, 0, strip=False, min_n=5, min_pur=1.00),
        "f3a2": build_k_lookup(raw_rows, y_raw, 3, True, 1, strip=False, min_n=10, min_pur=0.98),
        "f4a2_exact": build_k_lookup(raw_rows, y_raw, 4, True, 1, strip=False, min_n=5, min_pur=1.00),
        # end-date / factory rules
        "l2a45": build_k_lookup_pair(raw_rows, y_raw, 2, False, [3, 4], strip=True, min_n=10, min_pur=0.95),
        "l3a4_exact": build_k_lookup(raw_rows, y_raw, 3, False, 3, strip=True, min_n=5, min_pur=1.00),
        "l3a5_exact": build_k_lookup(raw_rows, y_raw, 3, False, 4, strip=True, min_n=5, min_pur=1.00),
        "l2a6": build_k_lookup(raw_rows, y_raw, 2, False, 5, strip=True, min_n=10, min_pur=0.98),
        "l3a6_exact": build_k_lookup(raw_rows, y_raw, 3, False, 5, strip=True, min_n=5, min_pur=1.00),
        # boss a6 rules
        "boss_l3a6_raw": build_k_lookup(raw_rows, y_raw, 3, False, 5, strip=False, min_n=3, min_pur=1.00, regime_only=2),
        "boss_l2a6_raw": build_k_lookup(raw_rows, y_raw, 2, False, 5, strip=False, min_n=3, min_pur=1.00, regime_only=2),
        # attr_3 hidden middle rules
        "mid_a3_pos45": build_pospair_lookup(raw_rows, y_raw, (4, 5), 2, rev=False, min_n=5, min_pur=1.00),
        "mid_a3_rev67": build_pospair_lookup(raw_rows, y_raw, (6, 7), 2, rev=True, min_n=5, min_pur=1.00),
        "mid_a3_pos45_mean": build_pospair_stats(raw_rows, y_raw, (4, 5), 2, rev=False, min_n=16, max_std=4.0),
        "mid_a3_rev67_mean": build_pospair_stats(raw_rows, y_raw, (6, 7), 2, rev=True, min_n=16, max_std=4.0),
    }


def apply_postproc(pred, raw_rows, lookups):
    out = pred.copy()
    hit_counter = defaultdict(int)

    for i, row in enumerate(raw_rows):
        if not row:
            continue

        row_t = tuple(row)
        if row_t in lookups["det_full"]:
            out[i, :] = np.array(lookups["det_full"][row_t], dtype=np.int64)
            hit_counter["det_full"] += 1
            continue

        pos = [j for j, t in enumerate(row) if t == SEP_TOKEN]
        is_boss = (len(pos) == 0) and (TOKEN_104 in row)
        stripped = strip104(row)
        L = len(row)

        # full split exact if both halves known
        if pos:
            bf = tuple(row[:pos[-1]])
            af = tuple(row[pos[-1] + 1:])
            if bf in lookups["det_b"] and af in lookups["det_a"]:
                vb = lookups["det_b"][bf]
                va = lookups["det_a"][af]
                out[i, 0:3] = np.array(vb, dtype=np.int64)
                out[i, 3:6] = np.array(va, dtype=np.int64)
                hit_counter["full_split"] += 1
                continue

        # attr1 exact-first
        if len(row) >= 3:
            k3 = tuple(row[:3])
            if k3 in lookups["f3a1_exact"]:
                out[i, 0] = int(lookups["f3a1_exact"][k3])
                hit_counter["f3a1_exact"] += 1
            elif len(row) >= 2:
                k2 = tuple(row[:2])
                if k2 in lookups["f2a1"]:
                    out[i, 0] = int(lookups["f2a1"][k2])
                    hit_counter["f2a1"] += 1
        elif len(row) >= 2:
            k2 = tuple(row[:2])
            if k2 in lookups["f2a1"]:
                out[i, 0] = int(lookups["f2a1"][k2])
                hit_counter["f2a1"] += 1

        # attr2 exact-first
        if len(row) >= 4:
            k4 = tuple(row[:4])
            if k4 in lookups["f4a2_exact"]:
                out[i, 1] = int(lookups["f4a2_exact"][k4])
                hit_counter["f4a2_exact"] += 1
            else:
                k3 = tuple(row[:3])
                if k3 in lookups["f3a2"]:
                    out[i, 1] = int(lookups["f3a2"][k3])
                    hit_counter["f3a2"] += 1
        elif len(row) >= 3:
            k3 = tuple(row[:3])
            if k3 in lookups["f3a2"]:
                out[i, 1] = int(lookups["f3a2"][k3])
                hit_counter["f3a2"] += 1

        # attr3 exact then soft memory
        a3_exact = False
        if L >= 8:
            rk = (row[-6], row[-7])
            if rk in lookups["mid_a3_rev67"]:
                out[i, 2] = int(lookups["mid_a3_rev67"][rk])
                hit_counter["mid_a3_rev67"] += 1
                a3_exact = True
        if (not a3_exact) and L >= 6:
            pk = (row[4], row[5])
            if pk in lookups["mid_a3_pos45"]:
                out[i, 2] = int(lookups["mid_a3_pos45"][pk])
                hit_counter["mid_a3_pos45"] += 1
                a3_exact = True

        if not a3_exact:
            if L >= 8:
                rk = (row[-6], row[-7])
                if rk in lookups["mid_a3_rev67_mean"]:
                    mu, n, _ = lookups["mid_a3_rev67_mean"][rk]
                    out[i, 2] = int(np.clip(round(0.75 * out[i, 2] + 0.25 * mu), 0, 99))
                    hit_counter["mid_a3_rev67_mean"] += 1
            if L >= 6:
                pk = (row[4], row[5])
                if pk in lookups["mid_a3_pos45_mean"]:
                    mu, n, _ = lookups["mid_a3_pos45_mean"][pk]
                    out[i, 2] = int(np.clip(round(0.75 * out[i, 2] + 0.25 * mu), 0, 99))
                    hit_counter["mid_a3_pos45_mean"] += 1

        # attr6 exact-first
        if is_boss:
            if len(row) >= 3:
                rk3 = tuple(row[-3:])
                if rk3 in lookups["boss_l3a6_raw"]:
                    out[i, 5] = int(lookups["boss_l3a6_raw"][rk3])
                    hit_counter["boss_l3a6_raw"] += 1
                elif len(row) >= 2:
                    rk2 = tuple(row[-2:])
                    if rk2 in lookups["boss_l2a6_raw"]:
                        out[i, 5] = int(lookups["boss_l2a6_raw"][rk2])
                        hit_counter["boss_l2a6_raw"] += 1
        else:
            if len(stripped) >= 3:
                sk3 = tuple(stripped[-3:])
                if sk3 in lookups["l3a6_exact"]:
                    out[i, 5] = int(lookups["l3a6_exact"][sk3])
                    hit_counter["l3a6_exact"] += 1
                elif len(stripped) >= 2:
                    sk2 = tuple(stripped[-2:])
                    if sk2 in lookups["l2a6"]:
                        out[i, 5] = int(lookups["l2a6"][sk2])
                        hit_counter["l2a6"] += 1
            elif len(stripped) >= 2:
                sk2 = tuple(stripped[-2:])
                if sk2 in lookups["l2a6"]:
                    out[i, 5] = int(lookups["l2a6"][sk2])
                    hit_counter["l2a6"] += 1

        # attr4/5 exact-first
        if len(stripped) >= 3:
            sk3 = tuple(stripped[-3:])
            hit45 = False
            if sk3 in lookups["l3a4_exact"]:
                out[i, 3] = int(lookups["l3a4_exact"][sk3])
                hit_counter["l3a4_exact"] += 1
                hit45 = True
            if sk3 in lookups["l3a5_exact"]:
                out[i, 4] = int(lookups["l3a5_exact"][sk3])
                hit_counter["l3a5_exact"] += 1
                hit45 = True
            if (not hit45) and len(stripped) >= 2:
                sk2 = tuple(stripped[-2:])
                if sk2 in lookups["l2a45"]:
                    v4, v5 = lookups["l2a45"][sk2]
                    out[i, 3] = int(v4)
                    out[i, 4] = int(v5)
                    hit_counter["l2a45"] += 1
        elif len(stripped) >= 2:
            sk2 = tuple(stripped[-2:])
            if sk2 in lookups["l2a45"]:
                v4, v5 = lookups["l2a45"][sk2]
                out[i, 3] = int(v4)
                out[i, 4] = int(v5)
                hit_counter["l2a45"] += 1

        # short-sequence business prior
        if L <= 5:
            out[i, 4] = out[i, 1]
            out[i, 3] = out[i, 0]
        elif L == 6:
            out[i, 4] = out[i, 1]

    return out, dict(sorted(hit_counter.items()))


# -------------------------------
# DATASET
# -------------------------------
class FastSeqDataset(Dataset):
    def __init__(self, enc, labels=None):
        self.tensors = {
            "tokens": torch.from_numpy(enc["tokens"]).long(),
            "sessions": torch.from_numpy(enc["sessions"]).long(),
            "splits": torch.from_numpy(enc["splits"]).long(),
            "positions": torch.from_numpy(enc["positions"]).long(),
            "rev_positions": torch.from_numpy(enc["rev_positions"]).long(),
            "mask": torch.from_numpy(enc["mask"]),
            "regime": torch.from_numpy(enc["regime"]).long(),
            "has103": torch.from_numpy(enc["has103"]).long(),
            "has104": torch.from_numpy(enc["has104"]).long(),
            "has609": torch.from_numpy(enc["has609"]).long(),
            "starts609": torch.from_numpy(enc["starts609"]).long(),
            "end104": torch.from_numpy(enc["end104"]).long(),
            "seq_len": torch.from_numpy(enc["seq_len"]).long(),
            "len_bucket": torch.from_numpy(enc["len_bucket"]).long(),
            "num103": torch.from_numpy(enc["num103"]).long(),
            "last_seg_len": torch.from_numpy(enc["last_seg_len"]).long(),
            "f2": torch.from_numpy(enc["f2"]).long(),
            "f3": torch.from_numpy(enc["f3"]).long(),
            "f4": torch.from_numpy(enc["f4"]).long(),
            "l2": torch.from_numpy(enc["l2"]).long(),
            "l3": torch.from_numpy(enc["l3"]).long(),
            "m2": torch.from_numpy(enc["m2"]).long(),
            "rm2": torch.from_numpy(enc["rm2"]).long(),
        }
        self.labels = None if labels is None else torch.from_numpy(labels).long()
        self.n = self.tensors["tokens"].size(0)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.tensors.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def make_loader(ds, batch_size, shuffle):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(ds, **kwargs)


# -------------------------------
# MODEL
# -------------------------------
class FastTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        z = self.ln1(x)
        a, _ = self.attn(z, z, z, key_padding_mask=pad_mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


def masked_mean_pool(x, mask):
    w = mask.float().unsqueeze(-1)
    s = (x * w).sum(dim=1)
    d = w.sum(dim=1).clamp(min=1.0)
    return s / d


def select_or_fallback(primary_mask, fallback_mask):
    empty = primary_mask.sum(dim=1) == 0
    return torch.where(empty.unsqueeze(1), fallback_mask, primary_mask)


def pick_regime_from_heads(heads, rep, regime):
    # heads: ModuleList of 3 heads, one per regime
    all_out = torch.stack([h(rep) for h in heads], dim=1)
    idx = torch.arange(rep.size(0), device=rep.device)
    return all_out[idx, regime]


class SeqModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        d = D_MODEL

        self.token_emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.pos_emb = nn.Embedding(SEQ_LEN, d)
        self.rev_pos_emb = nn.Embedding(SEQ_LEN, d)
        self.session_emb = nn.Embedding(MAX_SESSIONS, d)
        self.split_emb = nn.Embedding(2, d)
        self.regime_emb = nn.Embedding(3, d)

        self.flag_emb = nn.ModuleDict({
            "has103": nn.Embedding(2, 8),
            "has104": nn.Embedding(2, 8),
            "has609": nn.Embedding(2, 8),
            "starts609": nn.Embedding(2, 8),
            "end104": nn.Embedding(2, 8),
            "len_bucket": nn.Embedding(LEN_BUCKETS, 12),
            "f2": nn.Embedding(HASH2, 16),
            "f3": nn.Embedding(HASH3, 24),
            "f4": nn.Embedding(HASH3, 24),
            "l2": nn.Embedding(HASH2, 16),
            "l3": nn.Embedding(HASH3, 24),
            "m2": nn.Embedding(HASH2, 16),
            "rm2": nn.Embedding(HASH2, 16),
        })

        self.blocks = nn.ModuleList([FastTransformerBlock(d, N_HEADS, FF_DIM, DROPOUT) for _ in range(N_LAYERS)])
        self.ln = nn.LayerNorm(d)

        self.global_proj = nn.Linear(d, 96)
        self.head_proj = nn.Linear(d, 96)
        self.tail_proj = nn.Linear(d, 96)
        self.mid_proj = nn.Linear(d, 96)
        self.rmid_proj = nn.Linear(d, 96)

        self.meta_proj = nn.Sequential(
            nn.Linear(8 + 12 + 8 * 5, 96),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(96, 64),
            nn.GELU(),
        )
        self.prefix_proj = nn.Sequential(nn.Linear(16 + 24 + 24, 64), nn.GELU(), nn.Dropout(DROPOUT))
        self.suffix_proj = nn.Sequential(nn.Linear(16 + 24, 64), nn.GELU(), nn.Dropout(DROPOUT))
        self.midfeat_proj = nn.Sequential(nn.Linear(16 + 16, 48), nn.GELU(), nn.Dropout(DROPOUT))

        self.branch12 = nn.Sequential(nn.Linear(96 + 96 + 64 + 64, 192), nn.GELU(), nn.Dropout(DROPOUT))
        self.branch3 = nn.Sequential(nn.Linear(96 + 96 + 96 + 96 + 48 + 64, 224), nn.GELU(), nn.Dropout(DROPOUT))
        self.branch45 = nn.Sequential(nn.Linear(96 + 96 + 64 + 64, 192), nn.GELU(), nn.Dropout(DROPOUT))
        self.branch6 = nn.Sequential(nn.Linear(96 + 96 + 64 + 64, 192), nn.GELU(), nn.Dropout(DROPOUT))

        # light regime experts: small regime-specific residual adapters + separate heads per regime
        self.regime_resid12 = nn.Embedding(3, 192)
        self.regime_resid3  = nn.Embedding(3, 224)
        self.regime_resid45 = nn.Embedding(3, 192)
        self.regime_resid6  = nn.Embedding(3, 192)

        self.attr1_heads = nn.ModuleList([nn.Sequential(nn.Linear(192, 96), nn.GELU(), nn.Linear(96, 12)) for _ in range(3)])
        self.attr2_heads = nn.ModuleList([nn.Sequential(nn.Linear(192, 96), nn.GELU(), nn.Linear(96, 31)) for _ in range(3)])

        self.attr3_reg_heads = nn.ModuleList([nn.Sequential(nn.Linear(224, 96), nn.GELU(), nn.Linear(96, 1)) for _ in range(3)])
        self.attr3_cls_heads = nn.ModuleList([nn.Sequential(nn.Linear(224, 128), nn.GELU(), nn.Linear(128, 100)) for _ in range(3)])

        self.attr4_heads = nn.ModuleList([nn.Sequential(nn.Linear(192, 96), nn.GELU(), nn.Linear(96, 12)) for _ in range(3)])
        self.attr5_heads = nn.ModuleList([nn.Sequential(nn.Linear(192, 96), nn.GELU(), nn.Linear(96, 31)) for _ in range(3)])

        self.attr6_reg_heads = nn.ModuleList([nn.Sequential(nn.Linear(192, 96), nn.GELU(), nn.Linear(96, 1)) for _ in range(3)])
        self.attr6_cls_heads = nn.ModuleList([nn.Sequential(nn.Linear(192, 128), nn.GELU(), nn.Linear(128, 100)) for _ in range(3)])

    def forward(self, batch):
        tokens = batch["tokens"]
        mask = batch["mask"]
        pad_mask = ~mask

        x = (
            self.token_emb(tokens)
            + self.pos_emb(batch["positions"])
            + self.rev_pos_emb(batch["rev_positions"])
            + self.session_emb(batch["sessions"])
            + self.split_emb(batch["splits"])
            + self.regime_emb(batch["regime"]).unsqueeze(1)
        )

        for blk in self.blocks:
            x = blk(x, pad_mask)
        x = self.ln(x)

        valid = mask
        trimmed_valid = valid.clone()
        last_idx = valid.long().sum(dim=1) - 1
        row_idx = torch.arange(tokens.size(0), device=tokens.device)
        last_idx_safe = last_idx.clamp(min=0)
        last_tok = tokens[row_idx, last_idx_safe]
        drop_last_104 = (last_idx >= 0) & (last_tok == TOKEN_104)
        trimmed_valid[row_idx[drop_last_104], last_idx_safe[drop_last_104]] = False
        no_token = trimmed_valid.sum(dim=1) == 0
        trimmed_valid = torch.where(no_token.unsqueeze(1), valid, trimmed_valid)

        head_mask = valid & (batch["splits"] == 0)
        tail_mask = valid & (batch["splits"] == 1)
        head_mask = select_or_fallback(head_mask, trimmed_valid)
        tail_mask = select_or_fallback(tail_mask, trimmed_valid)

        head_sel = head_mask & (batch["positions"] < 3)
        head_sel = select_or_fallback(head_sel, head_mask)
        tail_sel = tail_mask & (batch["rev_positions"] < 3) & trimmed_valid
        tail_sel = select_or_fallback(tail_sel, tail_mask)

        mid_sel = valid & (batch["positions"] >= 3) & (batch["positions"] <= 5)
        mid_sel = select_or_fallback(mid_sel, trimmed_valid)
        rmid_sel = valid & (batch["rev_positions"] >= 6) & (batch["rev_positions"] <= 7)
        rmid_sel = select_or_fallback(rmid_sel, trimmed_valid)

        glob = self.global_proj(masked_mean_pool(x, trimmed_valid))
        head = self.head_proj(masked_mean_pool(x, head_sel))
        tail = self.tail_proj(masked_mean_pool(x, tail_sel))
        mid = self.mid_proj(masked_mean_pool(x, mid_sel))
        rmid = self.rmid_proj(masked_mean_pool(x, rmid_sel))

        meta_num = torch.stack([
            batch["seq_len"].float() / SEQ_LEN,
            batch["last_seg_len"].float() / SEQ_LEN,
            batch["num103"].float() / 6.0,
            batch["regime"].float() / 2.0,
            batch["has103"].float(),
            batch["has104"].float(),
            batch["starts609"].float(),
            batch["end104"].float(),
        ], dim=1)
        meta_cat = torch.cat([
            self.flag_emb["has103"](batch["has103"]),
            self.flag_emb["has104"](batch["has104"]),
            self.flag_emb["has609"](batch["has609"]),
            self.flag_emb["starts609"](batch["starts609"]),
            self.flag_emb["end104"](batch["end104"]),
            self.flag_emb["len_bucket"](batch["len_bucket"]),
        ], dim=1)
        meta = self.meta_proj(torch.cat([meta_num, meta_cat], dim=1))

        prefix = self.prefix_proj(torch.cat([
            self.flag_emb["f2"](batch["f2"]),
            self.flag_emb["f3"](batch["f3"]),
            self.flag_emb["f4"](batch["f4"]),
        ], dim=1))
        suffix = self.suffix_proj(torch.cat([
            self.flag_emb["l2"](batch["l2"]),
            self.flag_emb["l3"](batch["l3"]),
        ], dim=1))
        midfeat = self.midfeat_proj(torch.cat([
            self.flag_emb["m2"](batch["m2"]),
            self.flag_emb["rm2"](batch["rm2"]),
        ], dim=1))

        rep12 = self.branch12(torch.cat([head, glob, prefix, meta], dim=1))
        rep3 = self.branch3(torch.cat([mid, rmid, head, glob, midfeat, meta], dim=1))
        rep45 = self.branch45(torch.cat([tail, glob, suffix, meta], dim=1))
        rep6 = self.branch6(torch.cat([tail, glob, suffix, meta], dim=1))

        reg = batch["regime"]
        rep12 = rep12 + self.regime_resid12(reg)
        rep3 = rep3 + self.regime_resid3(reg)
        rep45 = rep45 + self.regime_resid45(reg)
        rep6 = rep6 + self.regime_resid6(reg)

        return {
            "attr1": pick_regime_from_heads(self.attr1_heads, rep12, reg),
            "attr2": pick_regime_from_heads(self.attr2_heads, rep12, reg),
            "attr3_reg": pick_regime_from_heads(self.attr3_reg_heads, rep3, reg).squeeze(-1),
            "attr3_cls": pick_regime_from_heads(self.attr3_cls_heads, rep3, reg),
            "attr4": pick_regime_from_heads(self.attr4_heads, rep45, reg),
            "attr5": pick_regime_from_heads(self.attr5_heads, rep45, reg),
            "attr6_reg": pick_regime_from_heads(self.attr6_reg_heads, rep6, reg).squeeze(-1),
            "attr6_cls": pick_regime_from_heads(self.attr6_cls_heads, rep6, reg),
        }


# -------------------------------
# LOSS / TRAIN
# -------------------------------
def expected_value_loss(logits, target, ncls, scale, start_at_one=True):
    probs = F.softmax(logits, dim=1)
    start = 1.0 if start_at_one else 0.0
    values = torch.arange(start, start + ncls, device=logits.device, dtype=torch.float32).unsqueeze(0)
    pred = (probs * values).sum(dim=1)
    true = target.float() + start
    return ((pred - true) / scale) ** 2


def soft_classification_loss(logits, target, sigma=1.5):
    # target in [0,99]
    num_classes = logits.size(1)
    values = torch.arange(num_classes, device=logits.device, dtype=torch.float32).unsqueeze(0)
    t = target.float().unsqueeze(1)
    dist2 = (values - t) ** 2
    soft = torch.exp(-dist2 / (2 * sigma * sigma))
    soft = soft / soft.sum(dim=1, keepdim=True)
    logp = F.log_softmax(logits, dim=1)
    return -(soft * logp).sum(dim=1)


def model_loss(out, batch, ep_idx=None, total_epochs=FINAL_EPOCHS):
    y = batch["labels"]
    seq_len = batch["seq_len"].float()
    is_short = (seq_len <= 5).float()
    is_boss = (batch["regime"] == 2).float()
    is_split = (batch["regime"] == 0).float()

    # Lighter sample reweighting: keep month/day alive instead of letting attr_3/6 dominate everything.
    w_cls = 1.0 + 0.35 * is_boss
    w_day = (1.0 + 0.10 * (1.0 - is_short)) * w_cls
    w_mon = (1.0 + 0.05 * (1.0 - is_short)) * w_cls

    ce1 = F.cross_entropy(out["attr1"], y[:, 0], reduction="none")
    ce2 = F.cross_entropy(out["attr2"], y[:, 1], reduction="none")
    ce4 = F.cross_entropy(out["attr4"], y[:, 3], reduction="none")
    ce5 = F.cross_entropy(out["attr5"], y[:, 4], reduction="none")
    ce = ((ce1 * w_mon).mean() + (ce2 * w_day).mean() + (ce4 * w_mon).mean() + (ce5 * w_day).mean()) / 4.0

    ev1 = expected_value_loss(out["attr1"], y[:, 0], 12, 12.0, start_at_one=True)
    ev2 = expected_value_loss(out["attr2"], y[:, 1], 31, 31.0, start_at_one=True)
    ev4 = expected_value_loss(out["attr4"], y[:, 3], 12, 12.0, start_at_one=True)
    ev5 = expected_value_loss(out["attr5"], y[:, 4], 31, 31.0, start_at_one=True)
    ev = ((ev1 * w_mon).mean() + (ev2 * w_day).mean() + (ev4 * w_mon).mean() + (ev5 * w_day).mean()) / 4.0

    # dual-head attr3
    pred3 = torch.sigmoid(out["attr3_reg"])
    true3 = y[:, 2].float() / 99.0
    regw3 = (1.0 + 0.80 * is_short) * (1.0 + 0.60 * is_boss) * (1.0 + 0.20 * is_split)
    reg3 = (((pred3 - true3) ** 2) * regw3).mean()
    cls3 = (soft_classification_loss(out["attr3_cls"], y[:, 2], sigma=1.4) * regw3).mean()
    ev3 = (expected_value_loss(out["attr3_cls"], y[:, 2], 100, 99.0, start_at_one=False) * regw3).mean()

    # dual-head attr6
    pred6 = torch.sigmoid(out["attr6_reg"])
    true6 = y[:, 5].float() / 99.0
    regw6 = (1.0 + 0.40 * is_short) * (1.0 + 0.90 * is_boss)
    reg6 = (((pred6 - true6) ** 2) * regw6).mean()
    cls6 = (soft_classification_loss(out["attr6_cls"], y[:, 5], sigma=1.2) * regw6).mean()
    ev6 = (expected_value_loss(out["attr6_cls"], y[:, 5], 100, 99.0, start_at_one=False) * regw6).mean()

    loss_3 = reg3 + 0.35 * cls3 + 0.20 * ev3
    loss_6 = reg6 + 0.35 * cls6 + 0.20 * ev6

    # curriculum: let model learn shared structure first, then lean more into attr_3/6 later.
    if ep_idx is None:
        w3, w6 = 12.0, 14.0
    else:
        cutoff = int(0.60 * total_epochs)
        if ep_idx < cutoff:
            w3, w6 = 8.0, 10.0
        else:
            w3, w6 = 14.0, 18.0

    loss = AUX_CE_WEIGHT * ce + AUX_EV_WEIGHT * ev + w3 * loss_3 + w6 * loss_6
    return loss, {
        "loss": float(loss.detach().item()),
        "ce": float(ce.detach().item()),
        "ev": float(ev.detach().item()),
        "reg3": float(reg3.detach().item()),
        "reg6": float(reg6.detach().item()),
        "cls3": float(cls3.detach().item()),
        "cls6": float(cls6.detach().item()),
        "w3": float(w3),
        "w6": float(w6),
    }

def move_batch(batch):
    return {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}


def build_optimizer(model):
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, fused=(DEVICE == "cuda"))
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    return opt


def get_snapshot_schedule(epochs):
    if SNAPSHOT_EPOCHS is not None:
        return sorted(set([e for e in SNAPSHOT_EPOCHS if 1 <= e <= epochs]))
    pts = [int(round(epochs * 0.70)), int(round(epochs * 0.85)), epochs]
    return sorted(set(max(1, min(epochs, p)) for p in pts))


def train_full(model, train_loader):
    optimizer = build_optimizer(model)
    total_steps = FINAL_EPOCHS * len(train_loader)
    warmup_steps = max(20, int(total_steps * WARMUP_PCT))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    snap_epochs = get_snapshot_schedule(FINAL_EPOCHS)
    snapshots = []

    print(f"Training | batch={TRAIN_BS} | epochs={FINAL_EPOCHS} | steps/epoch={len(train_loader)}")
    print(f"Snapshots at epochs: {snap_epochs}")

    for ep in range(1, FINAL_EPOCHS + 1):
        t0 = time.time()
        model.train()

        ep_loss = torch.tensor(0.0, device=DEVICE)
        ep_ce = torch.tensor(0.0, device=DEVICE)
        ep_ev = torch.tensor(0.0, device=DEVICE)
        ep_reg3 = torch.tensor(0.0, device=DEVICE)
        ep_reg6 = torch.tensor(0.0, device=DEVICE)
        ep_cls3 = torch.tensor(0.0, device=DEVICE)
        ep_cls6 = torch.tensor(0.0, device=DEVICE)
        n_steps = 0

        for batch in train_loader:
            batch = move_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                out = model(batch)
                loss, parts = model_loss(out, batch, ep_idx=ep-1, total_epochs=FINAL_EPOCHS)
            loss.backward()
            optimizer.step()
            scheduler.step()

            ep_loss += loss.detach()
            ep_ce += torch.tensor(parts["ce"], device=DEVICE)
            ep_ev += torch.tensor(parts["ev"], device=DEVICE)
            ep_reg3 += torch.tensor(parts["reg3"], device=DEVICE)
            ep_reg6 += torch.tensor(parts["reg6"], device=DEVICE)
            ep_cls3 += torch.tensor(parts["cls3"], device=DEVICE)
            ep_cls6 += torch.tensor(parts["cls6"], device=DEVICE)
            n_steps += 1

        dt = time.time() - t0
        print(
            f"ep{ep:03d} | loss={(ep_loss/n_steps).item():.5f} ce={(ep_ce/n_steps).item():.5f} "
            f"ev={(ep_ev/n_steps).item():.5f} r3={(ep_reg3/n_steps).item():.5f} "
            f"c3={(ep_cls3/n_steps).item():.5f} r6={(ep_reg6/n_steps).item():.5f} "
            f"c6={(ep_cls6/n_steps).item():.5f} w3={parts["w3"]:.1f} w6={parts["w6"]:.1f} | {dt:.1f}s"
        )

        if ep in snap_epochs:
            snapshots.append({
                "epoch": ep,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            })
            print(f"  -> saved snapshot at epoch {ep}")

    return model, snapshots


# -------------------------------
# INFERENCE / SAVE
# -------------------------------
def decode_month_day(logits):
    return logits.argmax(dim=1)


def decode_dual(reg_logits, cls_logits, reg_weight=0.35):
    reg = torch.sigmoid(reg_logits) * 99.0
    probs = F.softmax(cls_logits, dim=1)
    vals = torch.arange(100, device=cls_logits.device, dtype=torch.float32).unsqueeze(0)
    ev = (probs * vals).sum(dim=1)
    pred = (reg_weight * reg + (1.0 - reg_weight) * ev).round()
    return pred.clamp(0, 99).long()


@torch.no_grad()
def predict_from_model(model, loader):
    model.eval()
    preds = []
    for batch in loader:
        batch = move_batch(batch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16):
            out = model(batch)
        a1 = decode_month_day(out["attr1"])
        a2 = decode_month_day(out["attr2"])
        a4 = decode_month_day(out["attr4"])
        a5 = decode_month_day(out["attr5"])
        a3 = decode_dual(out["attr3_reg"], out["attr3_cls"], reg_weight=0.40)
        a6 = decode_dual(out["attr6_reg"], out["attr6_cls"], reg_weight=0.32)
        pred = torch.stack([a1, a2, a3, a4, a5, a6], dim=1).cpu().numpy()
        preds.append(pred)
    return decode_preds(np.concatenate(preds, axis=0))


@torch.no_grad()
def predict_logits_from_model(model, loader):
    model.eval()
    outs = {
        "attr1": [], "attr2": [], "attr4": [], "attr5": [],
        "a3_reg": [], "a3_cls": [], "a6_reg": [], "a6_cls": []
    }
    for batch in loader:
        batch = move_batch(batch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16):
            out = model(batch)
        outs["attr1"].append(out["attr1"].float().cpu().numpy())
        outs["attr2"].append(out["attr2"].float().cpu().numpy())
        outs["attr4"].append(out["attr4"].float().cpu().numpy())
        outs["attr5"].append(out["attr5"].float().cpu().numpy())
        outs["a3_reg"].append(out["attr3_reg"].float().cpu().numpy())
        outs["a3_cls"].append(out["attr3_cls"].float().cpu().numpy())
        outs["a6_reg"].append(out["attr6_reg"].float().cpu().numpy())
        outs["a6_cls"].append(out["attr6_cls"].float().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in outs.items()}


@torch.no_grad()
def predict_snapshot_ensemble(vocab_size, loader, snapshots):
    if len(snapshots) == 0:
        raise RuntimeError("No snapshots saved.")

    acc = None
    for snap in snapshots:
        print(f"  -> infer snapshot epoch {snap['epoch']}")
        m = SeqModel(vocab_size).to(DEVICE)
        m.load_state_dict(snap["state"])
        logits_this = predict_logits_from_model(m, loader)
        if acc is None:
            acc = {k: v.copy() for k, v in logits_this.items()}
        else:
            for k in acc:
                acc[k] += logits_this[k]
        del m, logits_this
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    k = float(len(snapshots))
    attr1 = torch.from_numpy(acc["attr1"] / k)
    attr2 = torch.from_numpy(acc["attr2"] / k)
    attr4 = torch.from_numpy(acc["attr4"] / k)
    attr5 = torch.from_numpy(acc["attr5"] / k)
    a3_reg = torch.from_numpy(acc["a3_reg"] / k)
    a3_cls = torch.from_numpy(acc["a3_cls"] / k)
    a6_reg = torch.from_numpy(acc["a6_reg"] / k)
    a6_cls = torch.from_numpy(acc["a6_cls"] / k)

    a1 = decode_month_day(attr1)
    a2 = decode_month_day(attr2)
    a4 = decode_month_day(attr4)
    a5 = decode_month_day(attr5)
    a3 = decode_dual(a3_reg, a3_cls, reg_weight=0.40)
    a6 = decode_dual(a6_reg, a6_cls, reg_weight=0.32)

    pred = torch.stack([a1, a2, a3, a4, a5, a6], dim=1).cpu().numpy()
    return decode_preds(pred)

def save_checkpoint(path, model, token2id, id2token, lookups, snapshots_meta):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {
            "D_MODEL": D_MODEL,
            "N_LAYERS": N_LAYERS,
            "N_HEADS": N_HEADS,
            "FF_DIM": FF_DIM,
            "DROPOUT": DROPOUT,
            "SEQ_LEN": SEQ_LEN,
            "TRAIN_BS": TRAIN_BS,
            "FINAL_EPOCHS": FINAL_EPOCHS,
        },
        "token2id": token2id,
        "id2token": id2token,
        "lookups_full": lookups,
        "snapshots_meta": snapshots_meta,
    }
    torch.save(ckpt, path)
    print(f"Saved checkpoint: {path}")


# -------------------------------
# MAIN
# -------------------------------
def main():
    seed_everything(SEED)

    print("Loading data...")
    x_full, y_full, x_test = load_csvs()
    print(f"Full train={len(x_full)} | Test={len(x_test)} | SeqLen={SEQ_LEN}")

    token2id, id2token = build_vocab(x_full, x_test)
    vocab_size = len(token2id) + 2
    print(f"Vocab size={vocab_size}")

    print("Encoding rows...")
    enc_full = encode_rows(x_full, token2id)
    enc_test = encode_rows(x_test, token2id)
    y_full_enc, y_full_raw = encode_targets(y_full)

    print("Building lookups...")
    lookups_full = build_lookups(enc_full["raw_rows"], y_full_raw)

    train_ds = FastSeqDataset(enc_full, y_full_enc)
    test_ds = FastSeqDataset(enc_test, None)
    train_loader = make_loader(train_ds, TRAIN_BS, shuffle=True)
    test_loader = make_loader(test_ds, PRED_BS, shuffle=False)

    model = SeqModel(vocab_size).to(DEVICE)
    if USE_COMPILE and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"compile skipped: {e}")

    model, snapshots = train_full(model, train_loader)

    # Use snapshot ensemble if available; else fallback to last model
    if len(snapshots) >= 2:
        print("Predicting test with snapshot ensemble...")
        raw_pred = predict_snapshot_ensemble(vocab_size, test_loader, snapshots)
        snapshots_meta = [s["epoch"] for s in snapshots]
    else:
        print("Predicting test with final model...")
        raw_pred = predict_from_model(model, test_loader)
        snapshots_meta = []

    post_pred, hit_cnt = apply_postproc(raw_pred, enc_test["raw_rows"], lookups_full)
    print(f"Postproc hits: {hit_cnt}")

    sub = pd.DataFrame({
        "id": enc_test["ids"],
        "attr_1": post_pred[:, 0].astype(np.uint16),
        "attr_2": post_pred[:, 1].astype(np.uint16),
        "attr_3": post_pred[:, 2].astype(np.uint16),
        "attr_4": post_pred[:, 3].astype(np.uint16),
        "attr_5": post_pred[:, 4].astype(np.uint16),
        "attr_6": post_pred[:, 5].astype(np.uint16),
    })
    sub.to_csv(SUB_NAME, index=False)
    print(f"Saved submission: {SUB_NAME}")

    save_checkpoint(CKPT_NAME, model, token2id, id2token, lookups_full, snapshots_meta)

    del train_loader, test_loader, train_ds, test_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
