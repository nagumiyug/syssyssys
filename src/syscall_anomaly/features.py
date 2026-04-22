from __future__ import annotations

from collections import Counter
from itertools import islice
from math import log2
from typing import Iterable

import numpy as np
import pandas as pd

from .schema import validate_columns

# 敏感信息常量
DESIGN_FILE_EXTS = {
    ".fcstd",
    ".step",
    ".stp",
    ".kicad_pcb",
    ".kicad_sch",
    ".kicad_pro",
    ".kicad_prl",
}

PROJECT_SOURCE_PATH_TYPES = {"freecad_project_source", "kicad_project_source"}

NOISE_SYSCALLS = {"mmap", "munmap", "mprotect"}



# 生成 N-gram 序列
def _ngrams(tokens: list[str], n: int) -> Iterable[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return zip(*(islice(tokens, i, None) for i in range(n)))


# 计算香农信息熵，当恶意脚本在进行“批量导出”或“全盘扫描”时，它的系统调用种类往往极其单一（比如疯狂 read 和 write），这会导致熵值骤降
def _safe_entropy(values: list[int]) -> float:
    total = sum(values)
    if total <= 0:
        return 0.0
    probs = [v / total for v in values if v > 0]
    return -sum(p * log2(p) for p in probs)


# 将绝对次数转化为频率
def _normalized_counter(counter: Counter[str], prefix: str) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {f"{prefix}{key}": value / total for key, value in counter.items()}


def _count_counter(counter: Counter[str], prefix: str) -> dict[str, float]:
    return {f"{prefix}{key}": float(value) for key, value in counter.items()}


# 将时间差（Delta Time）分桶
def _bucketize_gap_ms(gaps_ms: np.ndarray) -> Counter[str]:
    buckets = Counter()
    for gap in gaps_ms:
        if gap < 1:
            buckets["lt_1ms"] += 1
        elif gap < 10:
            buckets["1_10ms"] += 1
        elif gap < 100:
            buckets["10_100ms"] += 1
        else:
            buckets["ge_100ms"] += 1
    return buckets


# 截断和过滤
def _top_keys(counter: Counter[str], limit: int) -> set[str]:
    if limit <= 0:
        return set()
    return {key for key, _ in counter.most_common(limit)}


def _filter_counter(counter: Counter[str], allowed: set[str] | None) -> Counter[str]:
    if not allowed:
        return counter
    return Counter({key: value for key, value in counter.items() if key in allowed})


# 计算状态切换的频率
def _switch_ratio(values: list[str]) -> float:
    if len(values) < 2:
        return 0.0
    switches = sum(1 for prev, curr in zip(values[:-1], values[1:]) if prev != curr)
    return switches / max(len(values) - 1, 1)


# 全局词表生成器
# 遍历所有的训练数据，找出最常见、最具代表性的行为特征
# 如高频的系统调用、常见的执行序列等，并生成词典，防止维度爆炸
# 保留出现频率最高的前 N 个特征，把长尾的罕见噪音全部过滤
def infer_feature_vocab(
    events: pd.DataFrame,
    *,
    syscall_limit: int = 32,
    transition_limit: int = 64,
    bigram_limit: int = 64,
    trigram_limit: int = 64,
    context_value_limit: int = 16,
) -> dict[str, set[str]]:
    validate_columns(events.columns.tolist())
    df = events.copy()
    # 时序对齐
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["session_id", "timestamp", "tid"], kind="stable")
    df = df[~df["syscall_name"].isin(NOISE_SYSCALLS)]

    syscall_counter: Counter[str] = Counter()
    transition_counter: Counter[str] = Counter()
    bigram_counter: Counter[str] = Counter()
    trigram_counter: Counter[str] = Counter()
    op_counter: Counter[str] = Counter()
    path_type_counter: Counter[str] = Counter()
    ext_counter: Counter[str] = Counter()
    rc_counter: Counter[str] = Counter()
    
    # 统计全局高频行为
    for _, group in df.groupby("session_id", sort=False):
        syscalls = group["syscall_name"].astype(str).tolist()
        syscall_counter.update(syscalls)
        transition_counter.update(f"{prev}->{curr}" for prev, curr in zip(syscalls[:-1], syscalls[1:]))
        bigram_counter.update("__".join(tokens) for tokens in _ngrams(syscalls, 2))
        trigram_counter.update("__".join(tokens) for tokens in _ngrams(syscalls, 3))
        op_counter.update(group.get("op_category", pd.Series(dtype=str)).fillna("unknown").astype(str))
        path_type_counter.update(group.get("path_type", pd.Series(dtype=str)).fillna("unknown").astype(str))
        ext_counter.update(group.get("file_ext", pd.Series(dtype=str)).fillna("unknown").astype(str))
        rc_counter.update(group.get("return_code", pd.Series(dtype=str)).fillna("unknown").astype(str))

    return {
        "syscalls": _top_keys(syscall_counter, syscall_limit),
        "transitions": _top_keys(transition_counter, transition_limit),
        "bigrams": _top_keys(bigram_counter, bigram_limit),
        "trigrams": _top_keys(trigram_counter, trigram_limit),
        "ctx_ops": _top_keys(op_counter, context_value_limit),
        "ctx_path_types": _top_keys(path_type_counter, context_value_limit),
        "ctx_exts": _top_keys(ext_counter, context_value_limit),
        "ctx_rcs": _top_keys(rc_counter, context_value_limit),
    }


# 生成特征向量
def build_session_features(
    events: pd.DataFrame,
    include_context: bool,
    vocab: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    validate_columns(events.columns.tolist())
    df = events.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["session_id", "timestamp", "tid"], kind="stable")
    df = df[~df["syscall_name"].isin(NOISE_SYSCALLS)]

    rows: list[dict[str, object]] = []
    WINDOW_SIZE = 1000
    # for session_id, group in df.groupby("session_id", sort=False):
    for session_id, full_group in df.groupby("session_id", sort=False):
        full_group = full_group.reset_index(drop=True)
        # 第二层：在单个 Session 内部进行切片
        for chunk_start in range(0, len(full_group), WINDOW_SIZE):
            # 截取当前窗口的数据块
            group = full_group.iloc[chunk_start : chunk_start + WINDOW_SIZE]
            # 防污染
            if len(group) < WINDOW_SIZE * 0.5:
                continue
            # 为当前切片生成一个唯一的 ID
            chunk_idx = chunk_start // WINDOW_SIZE
            chunk_session_id = f"{session_id}_chunk_{chunk_idx}"
            syscalls = group["syscall_name"].astype(str).tolist()
            syscall_counts = Counter(syscalls)
            transition_counts = Counter(
                f"{prev}->{curr}" for prev, curr in zip(syscalls[:-1], syscalls[1:])
            )
            bigram_counts = Counter("__".join(tokens) for tokens in _ngrams(syscalls, 2))
            trigram_counts = Counter("__".join(tokens) for tokens in _ngrams(syscalls, 3))
            filtered_syscalls = _filter_counter(syscall_counts, None if vocab is None else vocab.get("syscalls"))
            filtered_transitions = _filter_counter(
                transition_counts, None if vocab is None else vocab.get("transitions")
            )
            filtered_bigrams = _filter_counter(bigram_counts, None if vocab is None else vocab.get("bigrams"))
            filtered_trigrams = _filter_counter(trigram_counts, None if vocab is None else vocab.get("trigrams"))
    
            feature_row: dict[str, object] = {
                "session_id": chunk_session_id,
                "software": group["software"].iloc[0],
                "scenario": group["scenario"].iloc[0],
                "label": group["label"].iloc[0],
                # "seq_length": float(len(syscalls)),
                # "unique_syscalls": float(len(syscall_counts)),
                "syscall_entropy": _safe_entropy(list(syscall_counts.values())),
                "transition_entropy": _safe_entropy(list(transition_counts.values())),
                "self_loop_ratio": (
                    sum(1 for prev, curr in zip(syscalls[:-1], syscalls[1:]) if prev == curr)
                    / max(len(syscalls) - 1, 1)
                ),
            }
            feature_row.update(_count_counter(filtered_syscalls, "freq_count::"))
            feature_row.update(_normalized_counter(filtered_syscalls, "freq_norm::"))
            feature_row.update(_normalized_counter(filtered_transitions, "trans_norm::"))
            feature_row.update(_normalized_counter(filtered_bigrams, "seq2_norm::"))
            feature_row.update(_normalized_counter(filtered_trigrams, "seq3_norm::"))
    
            if include_context:
                op_counter = Counter(group.get("op_category", pd.Series(dtype=str)).fillna("unknown"))
                path_type_counter = Counter(group.get("path_type", pd.Series(dtype=str)).fillna("unknown"))
                ext_counter = Counter(group.get("file_ext", pd.Series(dtype=str)).fillna("unknown"))
                rc_counter = Counter(group.get("return_code", pd.Series(dtype=str)).fillna("unknown").astype(str))
                path_types = group.get("path_type", pd.Series(dtype=str)).fillna("unknown").astype(str).tolist()
                file_exts = group.get("file_ext", pd.Series(dtype=str)).fillna("unknown").astype(str).tolist()
                total_events = float(len(group))
                read_events = float(op_counter.get("read", 0))
                write_events = float(op_counter.get("write", 0))
                filtered_ops = _filter_counter(op_counter, None if vocab is None else vocab.get("ctx_ops"))
                filtered_path_types = _filter_counter(
                    path_type_counter, None if vocab is None else vocab.get("ctx_path_types")
                )
                filtered_exts = _filter_counter(ext_counter, None if vocab is None else vocab.get("ctx_exts"))
                filtered_rcs = _filter_counter(rc_counter, None if vocab is None else vocab.get("ctx_rcs"))
    
                ts = group["timestamp"].dropna()
                gaps_ms = (
                    ts.diff().dt.total_seconds().dropna().to_numpy(dtype=float) * 1000.0
                    if len(ts) > 1
                    else np.array([], dtype=float)
                )
                gap_counter = _bucketize_gap_ms(gaps_ms)
    
                feature_row.update(_normalized_counter(filtered_ops, "ctx_op::"))
                feature_row.update(_normalized_counter(filtered_path_types, "ctx_path_type::"))
                feature_row.update(_normalized_counter(filtered_exts, "ctx_ext::"))
                feature_row.update(_normalized_counter(filtered_rcs, "ctx_rc::"))
                feature_row.update(_normalized_counter(gap_counter, "ctx_gap::"))
                feature_row["ctx_unique_threads"] = float(group["tid"].nunique())
                feature_row["ctx_unique_pids"] = float(group["pid"].nunique())
                feature_row["ctx_remote_event_ratio"] = float(
                    group.get("remote_addr", pd.Series(dtype=str)).notna().mean()
                    if "remote_addr" in group
                    else 0.0
                )
                feature_row["ctx_source_copy_target_ratio"] = float(
                    path_type_counter.get("source_copy_target", 0) / max(total_events, 1.0)
                )
                # feature_row["ctx_project_source_ratio"] = float(
                #    sum(path_type_counter.get(path_type, 0) for path_type in PROJECT_SOURCE_PATH_TYPES)
                # 3    / max(total_events, 1.0)
               # )
                feature_row["ctx_has_source_copy_target"] = 1.0 if path_type_counter.get("source_copy_target", 0) > 0 else 0.0
            
                feature_row["ctx_has_project_source"] = 1.0 if any(path_type_counter.get(pt, 0) > 0 for pt in PROJECT_SOURCE_PATH_TYPES) else 0.0
                feature_row["ctx_design_file_ratio"] = float(
                    sum(1 for ext in file_exts if ext in DESIGN_FILE_EXTS) / max(total_events, 1.0)
                )
                feature_row["ctx_read_write_balance"] = float(write_events / max(read_events, 1.0))
                feature_row["ctx_path_type_switch_ratio"] = _switch_ratio(path_types)
    
            rows.append(feature_row)

    feature_df = pd.DataFrame(rows).fillna(0.0)
    return feature_df
