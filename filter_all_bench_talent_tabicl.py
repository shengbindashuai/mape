#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TALENT 批量评测（支持多模型顺序评测；每个模型内部 8 卡并行；按模型名分别汇总 + 总表）。

【已按方案A改造：支持边训练边测试】
- 传入 --models_dir 时：脚本会持续轮询目录，发现“新出现且稳定写完”的 ckpt 就立刻评测
- 用 tested_set 记录已评测 ckpt，避免重复
- 文件稳定性检测：满足
    1) mtime 距当前超过 --stable_sec
    2) 连续两次扫描文件大小一致
- 结果仍写入 outdir/<model_tag>/talent_*.txt，并追加总表 outdir/all_models_summary.tsv

其他行为保持和你原版一致：
- 每个模型内部固定 8 卡并行（不足则按可用卡数）
- --models_dir 下仅保留“每100 step”模型（可通过 --step_mod 改）
- SKIP_REGRESSION 默认跳过回归（原逻辑保留）
"""

from __future__ import annotations
import os, json, time, logging, warnings, argparse
from pathlib import Path
from typing import Optional, Tuple, Union, List
import multiprocessing as mp
import re
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ===== 固定参数（按你的要求写死） =====
DEFAULT_MODEL_PATH = "/vast/users/guangyi.chen/causal_group/zijian.li/LDM/Orion-MSP/stage1/checkpoint/dir2/step-27250d.ckpt"
DEFAULT_DATA_ROOT  = "/vast/users/guangyi.chen/causal_group/zijian.li/LDM/datasets"
DEFAULT_OUTDIR     = "evaluation_results_fulltrain2"
FIXED_GPUS         = 8                    # 固定 8 卡
COERCE_NUMERIC     = True                 # 固定自动数值化
MERGE_VAL          = True                 # 如果存在验证集则并入训练集（见 run_on_gpu 实现）
SKIP_REGRESSION    = True                 # 固定跳过回归
CLASSIFICATION_TASKS = {'binclass', 'multiclass'}

# ---------------- 工具函数（与你原版一致） ----------------
def convert_features(X: np.ndarray, enabled: bool) -> np.ndarray:
    X = np.asarray(X)
    if not enabled:
        return X
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    df = pd.DataFrame(X)
    encoded = pd.DataFrame(index=df.index)
    for col in df.columns:
        series = df.iloc[:, col]
        numeric_series = pd.to_numeric(series, errors='coerce')
        if series.isna().equals(numeric_series.isna()):
            encoded[col] = numeric_series
        else:
            string_series = series.astype("string")
            codes, uniques = pd.factorize(string_series, sort=True)
            codes = codes.astype(np.int32)
            if (codes == -1).any():
                codes[codes == -1] = len(uniques)
            encoded[col] = codes
    return encoded.fillna(0).values.astype(np.float32)

def handle_missing_entries(X: np.ndarray, y: np.ndarray, *, context: str) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X); y = np.asarray(y)
    df = pd.DataFrame(X); y_series = pd.Series(y, index=df.index)
    drop_mask = pd.Series(False, index=df.index)
    for col in df.columns:
        series = df.iloc[:, col]
        numeric_series = pd.to_numeric(series, errors='coerce')
        if series.isna().equals(numeric_series.isna()):
            nan_mask = numeric_series.isna()
            if nan_mask.any():
                mean_value = float(numeric_series.mean(skipna=True))
                if np.isnan(mean_value): mean_value = 0.0
                df.iloc[:, col] = numeric_series.fillna(mean_value)
        else:
            nan_mask = series.isna()
            if nan_mask.any(): drop_mask |= nan_mask
    if drop_mask.any():
        drop_count = int(drop_mask.sum())
        df = df.loc[~drop_mask].copy()
        y_series = y_series.loc[df.index]
        logging.info("%s: 删除 %d 行包含字符串缺失值", context, drop_count)
    return df.values, y_series.values

def count_missing(values: np.ndarray) -> int:
    if values is None: return 0
    arr = np.asarray(values)
    if arr.dtype.kind in {"f", "c"}:
        return int(np.isnan(arr).sum())
    mask = pd.isna(pd.DataFrame(arr))
    return int(mask.values.sum())

def log_nan_presence(context: str, values: np.ndarray, *, dataset_id: str | None = None, missing_registry: set[str] | None = None) -> None:
    missing = count_missing(values)
    if missing:
        logging.warning(f"{context}: 原始数据包含 {missing} 个 NaN/缺失值")
        if dataset_id and missing_registry is not None:
            missing_registry.add(dataset_id)

def load_array(file_path: Path) -> np.ndarray:
    suffix = file_path.suffix.lower()
    if suffix in {'.npy', '.npz'}:
        try:
            arr = np.load(file_path, allow_pickle=False)
        except ValueError:
            arr = np.load(file_path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[list(arr.files)[0]]
        return np.asarray(arr)
    if suffix == '.parquet':
        return pd.read_parquet(file_path).values
    sep = '\t' if suffix == '.tsv' else None
    return pd.read_csv(file_path, sep=sep, header=None).values

def find_data_files(dataset_dir: Path):
    files = [p for p in dataset_dir.iterdir() if p.is_file()]
    lower = {p.name.lower(): p for p in files}
    def by_suffix(key: str):
        for name, p in lower.items():
            if name.endswith(key): return p
        return None

    n_train = by_suffix('n_train.npy'); c_train = by_suffix('c_train.npy'); y_train = by_suffix('y_train.npy')
    n_val   = by_suffix('n_val.npy');   c_val   = by_suffix('c_val.npy');   y_val   = by_suffix('y_val.npy')
    n_test  = by_suffix('n_test.npy');  c_test  = by_suffix('c_test.npy');  y_test  = by_suffix('y_test.npy')

    if y_train and y_test and (n_train or c_train) and (n_test or c_test):
        val_pair = None
        if y_val and (n_val or c_val):
            val_pair = (n_val, c_val, y_val)
        return (n_train, c_train, y_train), val_pair, (n_test, c_test, y_test)

    table_candidates = [p for p in files if p.suffix.lower() in {'.npy', '.npz', '.csv', '.tsv', '.parquet'}]
    if len(table_candidates) == 1:
        return table_candidates[0], None, None
    return None, None, None

def load_pair(X_path: Path, y_path: Path, context: str = "", coerce_numeric: bool = False,
              dataset_id: str | None = None, missing_registry: set[str] | None = None):
    X = load_array(X_path); y = load_array(y_path)
    log_nan_presence(f"{context or X_path.stem}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
    log_nan_presence(f"{context or X_path.stem}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)
    X = np.asarray(X); y = np.asarray(y)
    if X.ndim == 1: X = X.reshape(-1, 1)
    if y.ndim > 1:
        if y.shape[1] == 1: y = y.squeeze(1)
        elif y.shape[0] == 1: y = y.squeeze(0)
    X, y = handle_missing_entries(X, y, context=context or X_path.stem)
    X = convert_features(X, coerce_numeric)
    return X, y

def load_split(num_path: Optional[Path], cat_path: Optional[Path], y_path: Path,
               context: str = "", coerce_numeric: bool = False,
               dataset_id: str | None = None, missing_registry: set[str] | None = None):
    feats = []
    base = context or (num_path.stem if num_path else (cat_path.stem if cat_path else y_path.stem))
    if num_path:
        Xn = np.asarray(load_array(num_path))
        if Xn.ndim == 1: Xn = Xn.reshape(-1, 1)
        log_nan_presence(f"{base}-num_raw", Xn, dataset_id=dataset_id, missing_registry=missing_registry)
        feats.append(Xn)
    if cat_path:
        Xc = np.asarray(load_array(cat_path))
        if Xc.ndim == 1: Xc = Xc.reshape(-1, 1)
        log_nan_presence(f"{base}-cat_raw", Xc, dataset_id=dataset_id, missing_registry=missing_registry)
        feats.append(Xc)
    if not feats: raise ValueError("缺少数值/类别特征文件")
    n = feats[0].shape[0]
    for i, f in enumerate(feats):
        if f.shape[0] != n:
            raise ValueError(f"特征数量不一致: #{i} 有 {f.shape[0]} vs {n}")
    X = feats[0] if len(feats) == 1 else np.concatenate(feats, axis=1)
    log_nan_presence(f"{base}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
    y = np.asarray(load_array(y_path))
    log_nan_presence(f"{base}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)
    if y.ndim > 1:
        if y.shape[1] == 1: y = y.squeeze(1)
        elif y.shape[0] == 1: y = y.squeeze(0)
    X, y = handle_missing_entries(X, y, context=base)
    X = convert_features(X, coerce_numeric)
    return X, y

def load_table(file_path: Union[Path, Tuple], context: str = "", coerce_numeric: bool = False,
               dataset_id: str | None = None, missing_registry: set[str] | None = None) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(file_path, (tuple, list)):
        if len(file_path) == 2:
            Xp, yp = Path(file_path[0]), Path(file_path[1])
            return load_pair(Xp, yp, context=context, coerce_numeric=coerce_numeric,
                             dataset_id=dataset_id, missing_registry=missing_registry)
        if len(file_path) == 3:
            num_path, cat_path, y_path = file_path
            return load_split(Path(num_path) if num_path else None,
                              Path(cat_path) if cat_path else None,
                              Path(y_path),
                              context=context, coerce_numeric=coerce_numeric,
                              dataset_id=dataset_id, missing_registry=missing_registry)
        raise ValueError(f"Unsupported tuple for load_table: {file_path}")

    path: Path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix in {'.npy', '.npz'}:
        try:
            arr = np.load(path, allow_pickle=False)
        except ValueError:
            arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[list(arr.files)[0]]
        data = np.asarray(arr)
    elif suffix == '.parquet':
        data = pd.read_parquet(path).values
    else:
        sep = '\t' if suffix == '.tsv' else None
        data = pd.read_csv(path, sep=sep, header=None).values

    if data.ndim == 1:
        raise ValueError(f"Unsupported 1D data in {path}")

    log_target = context or str(path)
    log_nan_presence(f"{log_target}-raw", data, dataset_id=dataset_id, missing_registry=missing_registry)

    col0 = data[:, 0]
    try:
        uniques0 = np.unique(col0)
    except Exception:
        uniques0 = np.array([])

    if 0 < uniques0.size < max(2, data.shape[0] // 2):
        y = data[:, 0]; X = data[:, 1:]; which = 'first'
    else:
        y = data[:, -1]; X = data[:, :-1]; which = 'last'

    log_nan_presence(f"{log_target}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
    log_nan_presence(f"{log_target}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)

    X = np.asarray(pd.DataFrame(X).values); y = pd.Series(y).values
    X, y = handle_missing_entries(X, y, context=log_target)
    X = convert_features(X, coerce_numeric)
    logging.info(f"{log_target}: 使用单文件启发式拆分标签 (取 {which} 列)")
    return X, y

def load_dataset_info(dataset_dir: Path) -> Optional[dict]:
    p = dataset_dir / 'info.json'
    if not p.exists(): return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as exc:
        logging.warning(f"读取 {p} 失败: {exc}")
        return None

def summarize_task_types(dirs: List[Path]) -> None:
    counts = {'regression': 0, 'binclass': 0, 'multiclass': 0, 'unknown': 0}
    for d in dirs:
        info = load_dataset_info(d)
        t = (str(info.get('task_type', '')).lower() if info else '')
        if not t: counts['unknown'] += 1
        elif t in counts: counts['regression' if t=='regression' else t] += 1
        else: counts['unknown'] += 1
    logging.info("任务统计: regression=%d, binclass=%d, multiclass=%d, unknown=%d, 总计=%d",
                 counts['regression'], counts['binclass'], counts['multiclass'], counts['unknown'], len(dirs))

# ---------------- 子进程：单卡评测（回传到共享结果表） ----------------
def run_on_gpu(model_path: str, dirs: List[Path], gpu_physical_id: int, results_list):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_physical_id)
    try:
        import torch
        torch.cuda.set_device(0)
    except Exception:
        pass

    logging.info(f"[GPU {gpu_physical_id}] 启动，分配到 {len(dirs)} 个数据集")

    # from orion_msp.sklearn.classifier import OrionMSPClassifier
    # clf = OrionMSPClassifier(verbose=False, model_path=model_path)

    from tabicl.sklearn.classifier import TabICLClassifier
    clf = TabICLClassifier(verbose=False, model_path=model_path)

    missing_datasets: set[str] = set()

    for d in dirs:
        try:
            info = load_dataset_info(d)
            ttype = (str(info.get('task_type', '')).lower() if info else None)
            if SKIP_REGRESSION and ttype == 'regression':
                logging.info(f"[GPU {gpu_physical_id}] 跳过 {d.name}: 回归任务")
                continue
            train_path, val_path, test_path = find_data_files(d)
            if train_path is None and test_path is None:
                logging.info(f"[GPU {gpu_physical_id}] 跳过 {d.name}: 未识别数据文件")
                continue

            if train_path and test_path:
                X_train, y_train = load_table(train_path, context=f"{d.name}-train",
                                              coerce_numeric=COERCE_NUMERIC, dataset_id=d.name,
                                              missing_registry=missing_datasets)
                X_test, y_test   = load_table(test_path,  context=f"{d.name}-test",
                                              coerce_numeric=COERCE_NUMERIC, dataset_id=d.name,
                                              missing_registry=missing_datasets)
            else:
                logging.info(f"[GPU {gpu_physical_id}] {d.name}: 只有单文件，当前策略跳过（如需 80/20 可再开启）")
                continue

            if d.name in missing_datasets:
                logging.info(f"[GPU {gpu_physical_id}] 跳过 {d.name}: 原始数据包含缺失值（按策略跳过）")
                # continue

            # 如果存在 val，则将其并入训练集（按你的要求）
            val_count = 0
            if val_path is not None and MERGE_VAL:
                try:
                    X_val, y_val = load_table(val_path, context=f"{d.name}-val",
                                              coerce_numeric=COERCE_NUMERIC, dataset_id=d.name,
                                              missing_registry=missing_datasets)
                    if X_val.ndim == 3 and X_val.shape[1] == 1: X_val = X_val.squeeze(1)
                    X_val = X_val.astype(np.float32, copy=False)
                    try:
                        X_train = np.concatenate([X_train, X_val], axis=0)
                        y_train = np.concatenate([y_train, y_val], axis=0)
                        val_count = int(X_val.shape[0])
                        logging.info(f"[GPU {gpu_physical_id}] {d.name}: 已将 val({val_count}) 并入 train")
                    except Exception as exc:
                        logging.warning(f"[GPU {gpu_physical_id}] {d.name}: 合并 val 失败：{exc}")
                        val_count = 0
                except Exception as exc:
                    logging.warning(f"[GPU {gpu_physical_id}] 无法加载 val {d.name}: {exc}")
                    val_count = 0

            # 调整维度并转成 float32
            if X_train.ndim == 3 and X_train.shape[1] == 1: X_train = X_train.squeeze(1)
            if X_test.ndim  == 3 and X_test.shape[1]  == 1: X_test  = X_test.squeeze(1)
            X_train = X_train.astype(np.float32, copy=False)
            X_test  = X_test.astype(np.float32, copy=False)

            # 计算样本数与训练占比（train 包含 val 已合并的情况下）
            train_count = int(X_train.shape[0])
            test_count  = int(X_test.shape[0])
            total_count = train_count + test_count
            train_ratio = float(train_count / total_count) if total_count > 0 else float("nan")

            t0 = time.time()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = float(np.mean(y_pred == y_test))
            dt = time.time() - t0
            logging.info(f"[GPU {gpu_physical_id}] {d.name}: acc={acc:.4f}, time={dt:.2f}s, train_ratio={train_ratio:.4f}")

            # 结果项格式为 (dataset, acc, time_s, train_ratio)
            results_list.append((d.name, acc, dt, train_ratio))

        except Exception as e:
            logging.exception(f"[GPU {gpu_physical_id}] 评测失败 {d.name}: {e}")

# ---------------- 主流程：评测“单个模型” ----------------
# def evaluate_model(model_path: str, data_root: Path, outdir_root: Path) -> Tuple[str, int, float, float, float, float]:
#     """
#     返回: (model_tag, total_datasets, avg_acc, total_time, avg_time, avg_train_ratio)
#     并把结果写入 outdir_root/<model_tag>/talent_*.txt
#     """
#     model_tag = Path(model_path).stem
#     outdir = outdir_root / model_tag
#     # outdir.mkdir(parents=True, exist_ok=True)

#     dirs = [d for d in sorted(data_root.iterdir()) if d.is_dir()]
#     summarize_task_types(dirs)

#     try:
#         available_gpus = int(os.environ.get("NUM_GPUS", "0"))
#     except Exception:
#         available_gpus = 0

#     num_gpus = FIXED_GPUS
#     if available_gpus > 0:
#         num_gpus = min(FIXED_GPUS, available_gpus)
#     if num_gpus < FIXED_GPUS:
#         logging.info(f"检测到 {num_gpus} 张 GPU（少于固定 8 张），将按 {num_gpus} 张并行。")

#     shards: List[List[Path]] = [[] for _ in range(num_gpus)]
#     for i, d in enumerate(dirs):
#         shards[i % num_gpus].append(d)

#     ctx = mp.get_context("spawn")
#     with ctx.Manager() as manager:
#         results_list = manager.list()

#         procs = []
#         for gpu_id in range(num_gpus):
#             p = ctx.Process(
#                 target=run_on_gpu,
#                 args=(model_path, shards[gpu_id], gpu_id, results_list),
#                 daemon=False,
#             )
#             p.start()
#             procs.append(p)

#         for p in procs:
#             p.join()

#         results = list(results_list)
#         results.sort(key=lambda x: x[0])

#         detailed_path = outdir / "talent_detailed.txt"
#         summary_path  = outdir / "talent_summary.txt"

#         if results:
#             # with open(detailed_path, "w") as f:
#             #     f.write("dataset\taccuracy\ttime_s\ttrain_ratio\n")
#             #     for name, acc, dur, tr in results:
#             #         tr_str = f"{tr:.6f}" if tr == tr else "nan"
#             #         f.write(f"{name}\t{acc:.6f}\t{dur:.3f}\t{tr_str}\n")

#             total_time = sum(dur for _, _, dur, _ in results)
#             avg_time   = total_time / len(results)
#             avg_acc    = sum(acc for _, acc, _, _ in results) / len(results)
#             tr_values = [tr for _, _, _, tr in results if tr == tr]
#             avg_train_ratio = (sum(tr_values) / len(tr_values)) if tr_values else float("nan")

#             with open(summary_path, "w") as f:
#                 f.write(f"Model: {model_tag}\n")
#                 f.write(f"Total datasets: {len(results)}\n")
#                 f.write(f"Average accuracy: {avg_acc:.6f}\n")
#                 f.write(f"Total time s: {total_time:.3f}\n")
#                 f.write(f"Average time s: {avg_time:.3f}\n")
#                 f.write(f"Average train_ratio: {avg_train_ratio:.6f}\n")

#             logging.info(f"[{model_tag}] 汇总完成：{detailed_path} / {summary_path}")
#             return model_tag, len(results), avg_acc, total_time, avg_time, avg_train_ratio
#         else:
#             logging.info(f"[{model_tag}] 没有成功的评测结果。")
#             return model_tag, 0, float("nan"), 0.0, float("nan"), float("nan")
def evaluate_model(model_path: str, data_root: Path, outdir_root: Path) -> Tuple[str, int, float, float, float, float]:
    """
    返回: (model_tag, total_datasets, avg_acc, total_time, avg_time, avg_train_ratio)

    注意：按需求【不再写】outdir_root/<model_tag>/talent_*.txt，
    只由外层 append_master() 追加写 outdir_root/all_models_summary.tsv
    """
    model_tag = Path(model_path).stem

    dirs = [d for d in sorted(data_root.iterdir()) if d.is_dir()]
    summarize_task_types(dirs)

    try:
        available_gpus = int(os.environ.get("NUM_GPUS", "0"))
    except Exception:
        available_gpus = 0

    num_gpus = FIXED_GPUS
    if available_gpus > 0:
        num_gpus = min(FIXED_GPUS, available_gpus)
    if num_gpus < FIXED_GPUS:
        logging.info(f"检测到 {num_gpus} 张 GPU（少于固定 8 张），将按 {num_gpus} 张并行。")

    shards: List[List[Path]] = [[] for _ in range(num_gpus)]
    for i, d in enumerate(dirs):
        shards[i % num_gpus].append(d)

    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        results_list = manager.list()

        procs = []
        for gpu_id in range(num_gpus):
            p = ctx.Process(
                target=run_on_gpu,
                args=(model_path, shards[gpu_id], gpu_id, results_list),
                daemon=False,
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        results = list(results_list)

        if results:
            # results: (dataset, acc, time_s, train_ratio)
            total_time = sum(dur for _, _, dur, _ in results)
            avg_time   = total_time / len(results)
            avg_acc    = sum(acc for _, acc, _, _ in results) / len(results)
            tr_values  = [tr for _, _, _, tr in results if tr == tr]
            avg_train_ratio = (sum(tr_values) / len(tr_values)) if tr_values else float("nan")

            logging.info(
                f"[{model_tag}] 汇总完成：datasets={len(results)}, "
                f"avg_acc={avg_acc:.6f}, total_time={total_time:.3f}s, "
                f"avg_time={avg_time:.3f}s, avg_train_ratio={avg_train_ratio:.6f}"
            )

            return model_tag, len(results), avg_acc, total_time, avg_time, avg_train_ratio
        else:
            logging.info(f"[{model_tag}] 没有成功的评测结果。")
            return model_tag, 0, float("nan"), 0.0, float("nan"), float("nan")


# ---------------- 方案A：轮询 models_dir 发现新 ckpt 并评测 ----------------
def discover_ckpts(models_dir: Path, *, step_mod: int = 100) -> List[Path]:
    files = [p for p in models_dir.iterdir()
             if p.is_file() and p.suffix.lower() in {".ckpt", ".pt", ".pth"}]

    # 排序函数：数字优先（常为 step），否则按修改时间
    def sort_key(p: Path):
        nums = re.findall(r"\d+", p.stem)
        if nums:
            return (0, int(nums[-1]), p.stem)
        return (1, int(p.stat().st_mtime), p.stem)

    ordered = sorted(files, key=sort_key)

    # 过滤逻辑：仅保留每 step_mod step 的模型；无数字则保留
    filtered: List[Path] = []
    for p in ordered:
        nums = re.findall(r"\d+", p.stem)
        if nums:
            step = int(nums[-1])
            if step_mod <= 1 or (step % step_mod == 0):
                filtered.append(p)
        else:
            filtered.append(p)
    return filtered

def is_file_stable(p: Path, last_sizes: dict[str, int], *, stable_sec: float = 10.0) -> bool:
    """
    稳定性判定：
    1) mtime 距现在超过 stable_sec（避免刚写完/仍在写）
    2) 连续两次扫描 size 一致（更保险）
    """
    key = str(p)
    try:
        st = p.stat()
    except Exception:
        return False

    age = time.time() - st.st_mtime
    if age < stable_sec:
        return False

    sz = st.st_size
    if key in last_sizes and last_sizes[key] == sz:
        return True
    last_sizes[key] = sz
    return False

def ensure_master_header(master_path: Path) -> None:
    if not master_path.exists():
        master_path.parent.mkdir(parents=True, exist_ok=True)
        with open(master_path, "w") as f:
            f.write("model_name\ttotal_datasets\taverage_accuracy\ttotal_time_s\taverage_time_s\taverage_train_ratio\n")

def append_master(master_path: Path, model_tag: str, total: int, avg_acc: float, total_t: float, avg_t: float, avg_tr: float) -> None:
    avg_acc_str = f"{avg_acc:.6f}" if avg_acc == avg_acc else "nan"
    avg_t_str   = f"{avg_t:.3f}"   if avg_t == avg_t else "nan"
    avg_tr_str  = f"{avg_tr:.6f}"  if avg_tr == avg_tr else "nan"
    with open(master_path, "a") as f:
        f.write(f"{model_tag}\t{total}\t{avg_acc_str}\t{total_t:.3f}\t{avg_t_str}\t{avg_tr_str}\n")

def loop_eval_new_ckpts(models_dir: str, data_root: Path, outdir_root: Path,
                        *, poll_sec: float = 30.0,
                        stable_sec: float = 10.0,
                        idle_exit_sec: float | None = None,
                        step_mod: int = 100) -> Path:
    """
    持续扫描 models_dir，发现新 ckpt（满足 step_mod + 稳定性）就依次评测。
    idle_exit_sec:
      - None: 一直运行
      - 数值: 若超过该秒数没新模型则退出
    返回 master_path
    """
    md = Path(models_dir)
    if not md.exists():
        raise FileNotFoundError(f"--models_dir 不存在: {md}")

    master_path = outdir_root / "all_models_summary.tsv"
    ensure_master_header(master_path)

    tested: set[str] = set()
    last_sizes: dict[str, int] = {}
    last_new_time = time.time()

    logging.info("进入轮询模式：dir=%s, poll_sec=%.1f, stable_sec=%.1f, idle_exit_sec=%s, step_mod=%d",
                 str(md), poll_sec, stable_sec, str(idle_exit_sec), step_mod)

    # 为了让“size 连续两次一致”生效：第一次扫描只记录 size，不立刻评测
    while True:
        ckpts = discover_ckpts(md, step_mod=step_mod)

        candidates: List[Path] = []
        for p in ckpts:
            sp = str(p)
            if sp in tested:
                continue
            if is_file_stable(p, last_sizes, stable_sec=stable_sec):
                candidates.append(p)

        if candidates:
            logging.info("发现 %d 个新模型待评测：%s",
                         len(candidates), " -> ".join([c.stem for c in candidates]))

            for p in candidates:
                sp = str(p)
                tested.add(sp)

                t0 = time.perf_counter()
                model_tag, total, avg_acc, total_t, avg_t, avg_tr = evaluate_model(sp, data_root, outdir_root)
                append_master(master_path, model_tag, total, avg_acc, total_t, avg_t, avg_tr)
                logging.info(f"[{model_tag}] Done in {time.perf_counter()-t0:.2f}s")

            last_new_time = time.time()
        else:
            if idle_exit_sec is not None:
                idle = time.time() - last_new_time
                if idle > idle_exit_sec:
                    logging.info("超过 %.0fs 没有新模型，退出轮询。", idle_exit_sec)
                    break

            time.sleep(poll_sec)

    return master_path

# ---------------- CLI：单模型或“轮询多模型目录” ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=None,
                    help="单个模型 ckpt 路径（与 --models_dir 互斥）")
    ap.add_argument("--models_dir", type=str, default=None,
                    help="包含多个 *.ckpt 的目录；轮询发现新模型就评测（方案A）")
    ap.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)

    # 轮询相关参数（方案A新增）
    ap.add_argument("--poll_sec", type=float, default=30.0,
                    help="轮询目录间隔秒数（默认 30）")
    ap.add_argument("--stable_sec", type=float, default=10.0,
                    help="文件 mtime 距现在至少 stable_sec 才认为可能写完（默认 10）")
    ap.add_argument("--idle_exit_sec", type=float, default=3600,
                    help="超过该秒数没有新模型就退出；默认 None 一直运行")
    ap.add_argument("--step_mod", type=int, default=100,
                    help="仅保留 step%%step_mod==0 的模型（默认 100；<=1 表示不过滤）")
    return ap.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    args = parse_args()

    data_root = Path(args.data_root)
    outdir_root = Path(args.outdir)
    outdir_root.mkdir(parents=True, exist_ok=True)

    # 优先：轮询目录模式（方案A）
    if args.models_dir:
        master_path = loop_eval_new_ckpts(
            args.models_dir, data_root, outdir_root,
            poll_sec=float(args.poll_sec),
            stable_sec=float(args.stable_sec),
            idle_exit_sec=(float(args.idle_exit_sec) if args.idle_exit_sec is not None else None),
            step_mod=int(args.step_mod),
        )
        print("\n汇总总表：", master_path)

        # exit()
        return

    # 否则：单模型模式（保持原行为）
    model_path = args.model_path or DEFAULT_MODEL_PATH

    master_path = outdir_root / "all_models_summary.tsv"
    ensure_master_header(master_path)

    t0_all = time.perf_counter()
    t0 = time.perf_counter()
    model_tag, total, avg_acc, total_t, avg_t, avg_tr = evaluate_model(model_path, data_root, outdir_root)
    append_master(master_path, model_tag, total, avg_acc, total_t, avg_t, avg_tr)
    logging.info(f"[{model_tag}] Done in {time.perf_counter()-t0:.2f}s")

    logging.info(f"完成，总耗时 {time.perf_counter()-t0_all:.2f}s")
    print("\n汇总总表：", master_path)

if __name__ == "__main__":
    main()
