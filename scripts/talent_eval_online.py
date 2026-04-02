# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# TALENT batch evaluation with online checkpoint watching.

# Key behavior:
# - If `--models_dir` is provided, keep polling this directory.
# - Whenever a new checkpoint appears and becomes stable, evaluate it once.
# - Each model evaluation shards datasets across up to 8 GPUs in parallel.
# - Write per-model outputs:
#   - <outdir>/<model_tag>/talent_detailed.txt
#   - <outdir>/<model_tag>/talent_summary.txt
# - Append global summary:
#   - <outdir>/all_models_summary.tsv
# """

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Allow running the script directly from repo root without editable install.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

# ===== Fixed defaults =====
DEFAULT_MODEL_PATH = "/vast/users/guangyi.chen/causal_group/zijian.li/LDM/Orion-MSP/stage1/checkpoint/dir2/step-27250d.ckpt"
DEFAULT_DATA_ROOT = "/vast/users/guangyi.chen/causal_group/zijian.li/LDM/datasets"
DEFAULT_OUTDIR = "evaluation_results_fulltrain_stage_rope"
FIXED_GPUS = 8
COERCE_NUMERIC = True
MERGE_VAL = True
SKIP_REGRESSION = True


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
        numeric_series = pd.to_numeric(series, errors="coerce")
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
    X = np.asarray(X)
    y = np.asarray(y)
    df = pd.DataFrame(X)
    y_series = pd.Series(y, index=df.index)
    drop_mask = pd.Series(False, index=df.index)
    for col in df.columns:
        series = df.iloc[:, col]
        numeric_series = pd.to_numeric(series, errors="coerce")
        if series.isna().equals(numeric_series.isna()):
            nan_mask = numeric_series.isna()
            if nan_mask.any():
                mean_value = float(numeric_series.mean(skipna=True))
                if np.isnan(mean_value):
                    mean_value = 0.0
                df.iloc[:, col] = numeric_series.fillna(mean_value)
        else:
            nan_mask = series.isna()
            if nan_mask.any():
                drop_mask |= nan_mask

    if drop_mask.any():
        drop_count = int(drop_mask.sum())
        df = df.loc[~drop_mask].copy()
        y_series = y_series.loc[df.index]
        logging.info("%s: 删除 %d 行包含字符串缺失值", context, drop_count)
    return df.values, y_series.values


def count_missing(values: np.ndarray) -> int:
    if values is None:
        return 0
    arr = np.asarray(values)
    if arr.dtype.kind in {"f", "c"}:
        return int(np.isnan(arr).sum())
    mask = pd.isna(pd.DataFrame(arr))
    return int(mask.values.sum())


def log_nan_presence(
    context: str,
    values: np.ndarray,
    *,
    dataset_id: str | None = None,
    missing_registry: set[str] | None = None,
) -> None:
    missing = count_missing(values)
    if missing:
        logging.warning("%s: 原始数据包含 %d 个 NaN/缺失值", context, missing)
        if dataset_id and missing_registry is not None:
            missing_registry.add(dataset_id)


def load_array(file_path: Path) -> np.ndarray:
    suffix = file_path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        try:
            arr = np.load(file_path, allow_pickle=False)
        except ValueError:
            arr = np.load(file_path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[list(arr.files)[0]]
        return np.asarray(arr)
    if suffix == ".parquet":
        return pd.read_parquet(file_path).values
    sep = "\t" if suffix == ".tsv" else None
    return pd.read_csv(file_path, sep=sep, header=None).values


def find_data_files(dataset_dir: Path):
    files = [p for p in dataset_dir.iterdir() if p.is_file()]
    lower = {p.name.lower(): p for p in files}

    def by_suffix(key: str):
        for name, p in lower.items():
            if name.endswith(key):
                return p
        return None

    n_train = by_suffix("n_train.npy")
    c_train = by_suffix("c_train.npy")
    y_train = by_suffix("y_train.npy")
    n_val = by_suffix("n_val.npy")
    c_val = by_suffix("c_val.npy")
    y_val = by_suffix("y_val.npy")
    n_test = by_suffix("n_test.npy")
    c_test = by_suffix("c_test.npy")
    y_test = by_suffix("y_test.npy")

    if y_train and y_test and (n_train or c_train) and (n_test or c_test):
        val_pair = None
        if y_val and (n_val or c_val):
            val_pair = (n_val, c_val, y_val)
        return (n_train, c_train, y_train), val_pair, (n_test, c_test, y_test)

    table_candidates = [p for p in files if p.suffix.lower() in {".npy", ".npz", ".csv", ".tsv", ".parquet"}]
    if len(table_candidates) == 1:
        return table_candidates[0], None, None
    return None, None, None


def load_pair(
    X_path: Path,
    y_path: Path,
    context: str = "",
    coerce_numeric: bool = False,
    dataset_id: str | None = None,
    missing_registry: set[str] | None = None,
):
    X = load_array(X_path)
    y = load_array(y_path)
    log_nan_presence(f"{context or X_path.stem}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
    log_nan_presence(f"{context or X_path.stem}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.ndim > 1:
        if y.shape[1] == 1:
            y = y.squeeze(1)
        elif y.shape[0] == 1:
            y = y.squeeze(0)
    X, y = handle_missing_entries(X, y, context=context or X_path.stem)
    X = convert_features(X, coerce_numeric)
    return X, y


def load_split(
    num_path: Optional[Path],
    cat_path: Optional[Path],
    y_path: Path,
    context: str = "",
    coerce_numeric: bool = False,
    dataset_id: str | None = None,
    missing_registry: set[str] | None = None,
):
    feats = []
    base = context or (num_path.stem if num_path else (cat_path.stem if cat_path else y_path.stem))

    if num_path:
        Xn = np.asarray(load_array(num_path))
        if Xn.ndim == 1:
            Xn = Xn.reshape(-1, 1)
        log_nan_presence(f"{base}-num_raw", Xn, dataset_id=dataset_id, missing_registry=missing_registry)
        feats.append(Xn)
    if cat_path:
        Xc = np.asarray(load_array(cat_path))
        if Xc.ndim == 1:
            Xc = Xc.reshape(-1, 1)
        log_nan_presence(f"{base}-cat_raw", Xc, dataset_id=dataset_id, missing_registry=missing_registry)
        feats.append(Xc)
    if not feats:
        raise ValueError("缺少数值/类别特征文件")

    n = feats[0].shape[0]
    for i, f in enumerate(feats):
        if f.shape[0] != n:
            raise ValueError(f"特征数量不一致: #{i} 有 {f.shape[0]} vs {n}")

    X = feats[0] if len(feats) == 1 else np.concatenate(feats, axis=1)
    log_nan_presence(f"{base}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
    y = np.asarray(load_array(y_path))
    log_nan_presence(f"{base}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)
    if y.ndim > 1:
        if y.shape[1] == 1:
            y = y.squeeze(1)
        elif y.shape[0] == 1:
            y = y.squeeze(0)
    X, y = handle_missing_entries(X, y, context=base)
    X = convert_features(X, coerce_numeric)
    return X, y


def load_table(
    file_path: Union[Path, Tuple],
    context: str = "",
    coerce_numeric: bool = False,
    dataset_id: str | None = None,
    missing_registry: set[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(file_path, (tuple, list)):
        if len(file_path) == 2:
            Xp, yp = Path(file_path[0]), Path(file_path[1])
            return load_pair(
                Xp,
                yp,
                context=context,
                coerce_numeric=coerce_numeric,
                dataset_id=dataset_id,
                missing_registry=missing_registry,
            )
        if len(file_path) == 3:
            num_path, cat_path, y_path = file_path
            return load_split(
                Path(num_path) if num_path else None,
                Path(cat_path) if cat_path else None,
                Path(y_path),
                context=context,
                coerce_numeric=coerce_numeric,
                dataset_id=dataset_id,
                missing_registry=missing_registry,
            )
        raise ValueError(f"Unsupported tuple for load_table: {file_path}")

    path: Path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        try:
            arr = np.load(path, allow_pickle=False)
        except ValueError:
            arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[list(arr.files)[0]]
        data = np.asarray(arr)
    elif suffix == ".parquet":
        data = pd.read_parquet(path).values
    else:
        sep = "\t" if suffix == ".tsv" else None
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
        y = data[:, 0]
        X = data[:, 1:]
        which = "first"
    else:
        y = data[:, -1]
        X = data[:, :-1]
        which = "last"

    log_nan_presence(f"{log_target}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
    log_nan_presence(f"{log_target}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)

    X = np.asarray(pd.DataFrame(X).values)
    y = pd.Series(y).values
    X, y = handle_missing_entries(X, y, context=log_target)
    X = convert_features(X, coerce_numeric)
    logging.info("%s: 使用单文件启发式拆分标签 (取 %s 列)", log_target, which)
    return X, y


def load_dataset_info(dataset_dir: Path) -> Optional[dict]:
    p = dataset_dir / "info.json"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logging.warning("读取 %s 失败: %s", p, exc)
        return None


def summarize_task_types(dirs: List[Path]) -> None:
    counts = {"regression": 0, "binclass": 0, "multiclass": 0, "unknown": 0}
    for d in dirs:
        info = load_dataset_info(d)
        t = str(info.get("task_type", "")).lower() if info else ""
        if not t:
            counts["unknown"] += 1
        elif t in counts:
            counts["regression" if t == "regression" else t] += 1
        else:
            counts["unknown"] += 1
    logging.info(
        "任务统计: regression=%d, binclass=%d, multiclass=%d, unknown=%d, 总计=%d",
        counts["regression"],
        counts["binclass"],
        counts["multiclass"],
        counts["unknown"],
        len(dirs),
    )


def resolve_gpu_devices(max_gpus: int = FIXED_GPUS) -> List[str]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env:
        devices = [x.strip() for x in env.split(",") if x.strip()]
    else:
        try:
            import torch

            n = torch.cuda.device_count()
        except Exception:
            n = 0
        devices = [str(i) for i in range(n)] if n > 0 else [str(i) for i in range(max_gpus)]

    if len(devices) > max_gpus:
        devices = devices[:max_gpus]
    return devices


def build_classifier(
    model_path: str,
    n_estimators: int,
    batch_size: Optional[int],
    n_jobs: Optional[int],
    use_torch_compile: bool,
    torch_compile_mode: Optional[str],
    torch_compile_backend: Optional[str],
    torch_compile_fullgraph: bool,
    torch_compile_dynamic: Optional[bool],
):
    """Build classifier with backward-compatible kwargs across tabicl versions."""
    from tabicl.sklearn.classifier import TabICLClassifier

    kwargs = {
        "verbose": False,
        "model_path": model_path,
        "device": "cuda",
        "use_amp": True,
        "n_estimators": n_estimators,
    }
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if n_jobs is not None:
        kwargs["n_jobs"] = n_jobs
    try:
        sig = inspect.signature(TabICLClassifier.__init__)
        params = sig.parameters
        if "use_kv_cache" in params:
            kwargs["use_kv_cache"] = True
        if use_torch_compile and "use_torch_compile" in params:
            kwargs["use_torch_compile"] = True
            if "torch_compile_mode" in params:
                kwargs["torch_compile_mode"] = torch_compile_mode
            if "torch_compile_backend" in params:
                kwargs["torch_compile_backend"] = torch_compile_backend
            if "torch_compile_fullgraph" in params:
                kwargs["torch_compile_fullgraph"] = torch_compile_fullgraph
            if "torch_compile_dynamic" in params:
                kwargs["torch_compile_dynamic"] = torch_compile_dynamic
    except Exception:
        pass
    return TabICLClassifier(**kwargs)


def preload_model_once(clf, gpu_device: str) -> None:
    """Force one-time checkpoint load and prevent per-dataset reload in legacy classifier."""
    load_fn = getattr(clf, "_load_model", None)
    if not callable(load_fn):
        return

    t0 = time.perf_counter()
    load_fn()

    # Newer classifier variants use _loaded_model_key in _ensure_model_loaded.
    if hasattr(clf, "_get_model_load_key"):
        try:
            clf._loaded_model_key = clf._get_model_load_key()
        except Exception:
            pass

    # Legacy classifier calls _load_model() in every fit(); make it a no-op after preload.
    def _skip_reloading():
        return None

    try:
        clf._load_model = _skip_reloading
    except Exception:
        pass

    logging.info("[GPU %s] 模型预加载完成: %.2fs", gpu_device, time.perf_counter() - t0)


def _set_cpu_thread_limits(cpu_threads: int) -> None:
    cpu_threads = max(1, int(cpu_threads))
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_threads)

    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=cpu_threads)
    except Exception:
        pass


def _dataset_cache_file(dataset_dir: Path, cache_root: Path) -> Path:
    dataset_key = hashlib.md5(str(dataset_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    return cache_root / f"{dataset_dir.name}__{dataset_key}.npz"


def _load_cached_dataset(cache_file: Path):
    try:
        with np.load(cache_file, allow_pickle=True) as z:
            X_train = z["X_train"]
            y_train = z["y_train"]
            X_test = z["X_test"]
            y_test = z["y_test"]
            train_ratio = float(z["train_ratio"])
        return X_train, y_train, X_test, y_test, train_ratio
    except Exception:
        return None


def _prepare_dataset_payload(dataset_dir: Path, cache_root: Path):
    info = load_dataset_info(dataset_dir)
    ttype = str(info.get("task_type", "")).lower() if info else None
    if SKIP_REGRESSION and ttype == "regression":
        return None, "回归任务", False

    cache_file = _dataset_cache_file(dataset_dir, cache_root)
    if cache_file.exists():
        cached = _load_cached_dataset(cache_file)
        if cached is not None:
            X_train, y_train, X_test, y_test, train_ratio = cached
            return (
                {
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_test": X_test,
                    "y_test": y_test,
                    "train_ratio": train_ratio,
                },
                None,
                True,
            )
        logging.warning("%s: 缓存文件损坏，重新构建: %s", dataset_dir.name, cache_file)

    missing_datasets: set[str] = set()
    train_path, val_path, test_path = find_data_files(dataset_dir)
    if train_path is None and test_path is None:
        return None, "未识别数据文件", False
    if not (train_path and test_path):
        return None, "只有单文件，当前策略跳过", False

    X_train, y_train = load_table(
        train_path,
        context=f"{dataset_dir.name}-train",
        coerce_numeric=COERCE_NUMERIC,
        dataset_id=dataset_dir.name,
        missing_registry=missing_datasets,
    )
    X_test, y_test = load_table(
        test_path,
        context=f"{dataset_dir.name}-test",
        coerce_numeric=COERCE_NUMERIC,
        dataset_id=dataset_dir.name,
        missing_registry=missing_datasets,
    )

    if dataset_dir.name in missing_datasets:
        logging.info("%s: 原始数据包含缺失值", dataset_dir.name)

    if val_path is not None and MERGE_VAL:
        try:
            X_val, y_val = load_table(
                val_path,
                context=f"{dataset_dir.name}-val",
                coerce_numeric=COERCE_NUMERIC,
                dataset_id=dataset_dir.name,
                missing_registry=missing_datasets,
            )
            if X_val.ndim == 3 and X_val.shape[1] == 1:
                X_val = X_val.squeeze(1)
            X_val = X_val.astype(np.float32, copy=False)
            X_train = np.concatenate([X_train, X_val], axis=0)
            y_train = np.concatenate([y_train, y_val], axis=0)
            logging.info("%s: 已将 val(%d) 并入 train", dataset_dir.name, int(X_val.shape[0]))
        except Exception as exc:
            logging.warning("%s: 合并 val 失败：%s", dataset_dir.name, exc)

    if X_train.ndim == 3 and X_train.shape[1] == 1:
        X_train = X_train.squeeze(1)
    if X_test.ndim == 3 and X_test.shape[1] == 1:
        X_test = X_test.squeeze(1)
    X_train = X_train.astype(np.float32, copy=False)
    X_test = X_test.astype(np.float32, copy=False)

    train_count = int(X_train.shape[0])
    test_count = int(X_test.shape[0])
    total_count = train_count + test_count
    train_ratio = float(train_count / total_count) if total_count > 0 else float("nan")

    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        np.savez(
            cache_file,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            train_ratio=np.array(train_ratio, dtype=np.float32),
        )
    except Exception as exc:
        logging.warning("%s: 写缓存失败 (%s): %s", dataset_dir.name, cache_file, exc)

    return (
        {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "train_ratio": train_ratio,
        },
        None,
        False,
    )


def _worker_loop(
    task_queue,
    result_queue,
    gpu_device: str,
    cache_root: str,
    clf_n_estimators: int,
    clf_batch_size: Optional[int],
    clf_n_jobs: Optional[int],
    cpu_threads: int,
    use_torch_compile: bool,
    torch_compile_mode: Optional[str],
    torch_compile_backend: Optional[str],
    torch_compile_fullgraph: bool,
    torch_compile_dynamic: Optional[bool],
    torchinductor_cache_dir: Optional[str],
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    if torchinductor_cache_dir:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = torchinductor_cache_dir
    _set_cpu_thread_limits(cpu_threads)
    try:
        import torch

        torch.cuda.set_device(0)
        torch.set_num_threads(max(1, int(cpu_threads)))
    except Exception:
        pass

    logging.info("[GPU %s] 启动，常驻worker模式", gpu_device)

    clf = None
    current_model_path = None
    cache_root_path = Path(cache_root)

    while True:
        task = task_queue.get()
        if task is None:
            break

        eval_id = int(task["eval_id"])
        model_path = str(task["model_path"])
        d = Path(task["dataset_dir"])

        if clf is None or current_model_path != model_path:
            clf = build_classifier(
                model_path=model_path,
                n_estimators=clf_n_estimators,
                batch_size=clf_batch_size,
                n_jobs=clf_n_jobs,
                use_torch_compile=use_torch_compile,
                torch_compile_mode=torch_compile_mode,
                torch_compile_backend=torch_compile_backend,
                torch_compile_fullgraph=torch_compile_fullgraph,
                torch_compile_dynamic=torch_compile_dynamic,
            )
            preload_model_once(clf, gpu_device)
            current_model_path = model_path

        ds_t0 = time.perf_counter()
        try:
            payload, skip_reason, cache_hit = _prepare_dataset_payload(d, cache_root_path)
            if payload is None:
                result_queue.put(
                    {
                        "eval_id": eval_id,
                        "status": "skip",
                        "dataset": d.name,
                        "reason": skip_reason,
                    }
                )
                continue

            X_train = payload["X_train"]
            y_train = payload["y_train"]
            X_test = payload["X_test"]
            y_test = payload["y_test"]
            train_ratio = float(payload["train_ratio"])

            prep_time = time.perf_counter() - ds_t0

            t_fit = time.perf_counter()
            clf.fit(X_train, y_train)
            fit_time = time.perf_counter() - t_fit

            t_pred = time.perf_counter()
            y_pred = clf.predict(X_test)
            pred_time = time.perf_counter() - t_pred

            acc = float(np.mean(y_pred == y_test))
            infer_time = fit_time + pred_time
            total_e2e = time.perf_counter() - ds_t0
            logging.info(
                "[GPU %s] %s: acc=%.4f, infer=%.2fs (fit=%.2fs, pred=%.2fs), prep=%.2fs, e2e=%.2fs, train_ratio=%.4f",
                gpu_device,
                d.name,
                acc,
                infer_time,
                fit_time,
                pred_time,
                prep_time,
                total_e2e,
                train_ratio,
            )
            result_queue.put(
                {
                    "eval_id": eval_id,
                    "status": "ok",
                    "dataset": d.name,
                    "acc": acc,
                    "infer_time": infer_time,
                    "train_ratio": train_ratio,
                    "prep_time": prep_time,
                    "fit_time": fit_time,
                    "pred_time": pred_time,
                    "total_e2e": total_e2e,
                    "cache_hit": cache_hit,
                }
            )

        except Exception as exc:
            logging.exception("[GPU %s] 评测失败 %s: %s", gpu_device, d.name, exc)
            result_queue.put(
                {
                    "eval_id": eval_id,
                    "status": "error",
                    "dataset": d.name,
                    "error": str(exc),
                }
            )


class PersistentEvaluatorPool:
    def __init__(
        self,
        gpu_devices: List[str],
        cache_root: Path,
        clf_n_estimators: int,
        clf_batch_size: Optional[int],
        clf_n_jobs: Optional[int],
        cpu_threads: int,
        use_torch_compile: bool,
        torch_compile_mode: Optional[str],
        torch_compile_backend: Optional[str],
        torch_compile_fullgraph: bool,
        torch_compile_dynamic: Optional[bool],
        torchinductor_cache_dir: Optional[str],
    ) -> None:
        if not gpu_devices:
            raise RuntimeError("未检测到可用GPU设备，无法执行并行评测。")

        self.ctx = mp.get_context("spawn")
        self.task_queue = self.ctx.Queue(maxsize=max(64, len(gpu_devices) * 8))
        self.result_queue = self.ctx.Queue()
        self.processes = []
        self.eval_id = 0
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)

        for gpu_dev in gpu_devices:
            p = self.ctx.Process(
                target=_worker_loop,
                args=(
                    self.task_queue,
                    self.result_queue,
                    gpu_dev,
                    str(self.cache_root),
                    clf_n_estimators,
                    clf_batch_size,
                    clf_n_jobs,
                    cpu_threads,
                    use_torch_compile,
                    torch_compile_mode,
                    torch_compile_backend,
                    torch_compile_fullgraph,
                    torch_compile_dynamic,
                    torchinductor_cache_dir,
                ),
                daemon=False,
            )
            p.start()
            self.processes.append(p)

        logging.info("常驻评测池已启动: %d workers, cache=%s", len(self.processes), self.cache_root)

    def close(self) -> None:
        for _ in self.processes:
            self.task_queue.put(None)
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)

    def evaluate_model(self, model_path: str, dirs: List[Path]):
        eval_id = self.eval_id
        self.eval_id += 1
        total_tasks = len(dirs)

        for d in dirs:
            self.task_queue.put({"eval_id": eval_id, "model_path": model_path, "dataset_dir": str(d)})

        results = []
        done = 0
        skip_count = 0
        err_count = 0
        cache_hits = 0
        while done < total_tasks:
            msg = self.result_queue.get()
            if int(msg.get("eval_id", -1)) != eval_id:
                continue
            done += 1
            status = msg.get("status")
            if status == "ok":
                results.append(
                    (
                        msg["dataset"],
                        float(msg["acc"]),
                        float(msg["infer_time"]),
                        float(msg["train_ratio"]),
                        float(msg["prep_time"]),
                        float(msg["fit_time"]),
                        float(msg["pred_time"]),
                        float(msg["total_e2e"]),
                    )
                )
                if msg.get("cache_hit"):
                    cache_hits += 1
            elif status == "skip":
                skip_count += 1
            else:
                err_count += 1
                logging.warning("数据集失败: %s (%s)", msg.get("dataset"), msg.get("error"))

        logging.info(
            "模型 %s 评测完成: ok=%d, skip=%d, error=%d, cache_hit=%d/%d",
            Path(model_path).stem,
            len(results),
            skip_count,
            err_count,
            cache_hits,
            total_tasks,
        )
        return results


def evaluate_model(
    model_path: str,
    outdir_root: Path,
    evaluator_pool: PersistentEvaluatorPool,
    dirs: List[Path],
) -> Tuple[str, int, float, float, float, float]:
    """
    Returns:
    (model_tag, total_datasets, avg_acc, total_time, avg_time, avg_train_ratio)
    """
    model_tag = Path(model_path).stem
    outdir = outdir_root / model_tag
    outdir.mkdir(parents=True, exist_ok=True)

    results = evaluator_pool.evaluate_model(model_path=model_path, dirs=dirs)
    results.sort(key=lambda x: x[0])

    detailed_path = outdir / "talent_detailed.txt"
    summary_path = outdir / "talent_summary.txt"

    if results:
        with open(detailed_path, "w") as f:
            f.write("dataset\taccuracy\ttime_s\ttrain_ratio\tprep_s\tfit_s\tpredict_s\ttotal_e2e_s\n")
            for name, acc, dur, tr, prep_s, fit_s, pred_s, e2e_s in results:
                tr_str = f"{tr:.6f}" if tr == tr else "nan"
                f.write(f"{name}\t{acc:.6f}\t{dur:.3f}\t{tr_str}\t{prep_s:.3f}\t{fit_s:.3f}\t{pred_s:.3f}\t{e2e_s:.3f}\n")

        total_time = sum(dur for _, _, dur, _, _, _, _, _ in results)
        avg_time = total_time / len(results)
        avg_acc = sum(acc for _, acc, _, _, _, _, _, _ in results) / len(results)
        tr_values = [tr for _, _, _, tr, _, _, _, _ in results if tr == tr]
        avg_prep_time = sum(prep for _, _, _, _, prep, _, _, _ in results) / len(results)
        avg_fit_time = sum(fit for _, _, _, _, _, fit, _, _ in results) / len(results)
        avg_pred_time = sum(pred for _, _, _, _, _, _, pred, _ in results) / len(results)
        avg_e2e_time = sum(e2e for _, _, _, _, _, _, _, e2e in results) / len(results)
        avg_train_ratio = (sum(tr_values) / len(tr_values)) if tr_values else float("nan")

        with open(summary_path, "w") as f:
            f.write(f"Model: {model_tag}\n")
            f.write(f"Total datasets: {len(results)}\n")
            f.write(f"Average accuracy: {avg_acc:.6f}\n")
            f.write(f"Total time s: {total_time:.3f}\n")
            f.write(f"Average time s: {avg_time:.3f}\n")
            f.write(f"Average prep time s: {avg_prep_time:.3f}\n")
            f.write(f"Average fit time s: {avg_fit_time:.3f}\n")
            f.write(f"Average predict time s: {avg_pred_time:.3f}\n")
            f.write(f"Average end-to-end time s: {avg_e2e_time:.3f}\n")
            f.write(f"Average train_ratio: {avg_train_ratio:.6f}\n")

        logging.info("[%s] 汇总完成：%s / %s", model_tag, detailed_path, summary_path)
        return model_tag, len(results), avg_acc, total_time, avg_time, avg_train_ratio

    logging.info("[%s] 没有成功的评测结果。", model_tag)
    return model_tag, 0, float("nan"), 0.0, float("nan"), float("nan")


def _extract_last_int(stem: str) -> Optional[int]:
    nums = re.findall(r"\d+", stem)
    if not nums:
        return None
    return int(nums[-1])


def discover_ckpts(models_dir: Path, step_mod: int) -> List[Path]:
    files = [p for p in models_dir.iterdir() if p.is_file() and p.suffix.lower() in {".ckpt", ".pt", ".pth"}]

    def sort_key(p: Path):
        step = _extract_last_int(p.stem)
        if step is not None:
            return (0, step, p.stem)
        return (1, int(p.stat().st_mtime), p.stem)

    ordered = sorted(files, key=sort_key)
    if step_mod <= 1:
        return ordered

    filtered = []
    for p in ordered:
        step = _extract_last_int(p.stem)
        if step is None or step % step_mod == 0:
            filtered.append(p)
    return filtered


def is_file_stable(p: Path, last_sizes: dict[str, int], stable_sec: float) -> bool:
    key = str(p)
    try:
        st = p.stat()
    except Exception:
        return False

    age = time.time() - st.st_mtime
    if age < stable_sec:
        return False

    size = st.st_size
    prev = last_sizes.get(key)
    last_sizes[key] = size
    return prev is not None and prev == size


def ensure_master_header(master_path: Path) -> None:
    if not master_path.exists():
        master_path.parent.mkdir(parents=True, exist_ok=True)
        with open(master_path, "w") as f:
            f.write("model_name\ttotal_datasets\taverage_accuracy\ttotal_time_s\taverage_time_s\taverage_train_ratio\n")


def append_master(master_path: Path, model_tag: str, total: int, avg_acc: float, total_t: float, avg_t: float, avg_tr: float):
    avg_acc_str = f"{avg_acc:.6f}" if avg_acc == avg_acc else "nan"
    avg_t_str = f"{avg_t:.3f}" if avg_t == avg_t else "nan"
    avg_tr_str = f"{avg_tr:.6f}" if avg_tr == avg_tr else "nan"
    with open(master_path, "a") as f:
        f.write(f"{model_tag}\t{total}\t{avg_acc_str}\t{total_t:.3f}\t{avg_t_str}\t{avg_tr_str}\n")


def load_tested(tested_log: Path) -> set[str]:
    if not tested_log.exists():
        return set()
    with open(tested_log, "r", encoding="utf-8") as f:
        return {str(Path(line.strip()).resolve()) for line in f if line.strip()}


def append_tested(tested_log: Path, ckpt_path: str) -> None:
    tested_log.parent.mkdir(parents=True, exist_ok=True)
    with open(tested_log, "a", encoding="utf-8") as f:
        f.write(str(Path(ckpt_path).resolve()) + "\n")


def loop_eval_new_ckpts(
    models_dir: Path,
    outdir_root: Path,
    poll_sec: float,
    stable_sec: float,
    idle_exit_sec: Optional[float],
    step_mod: int,
    tested_log: Path,
    evaluator_pool: PersistentEvaluatorPool,
    dirs: List[Path],
) -> Path:
    master_path = outdir_root / "all_models_summary.tsv"
    ensure_master_header(master_path)
    tested = load_tested(tested_log)
    last_sizes: dict[str, int] = {}
    last_new_ts = time.time()

    logging.info(
        "进入在线评测模式: models_dir=%s, poll_sec=%.1f, stable_sec=%.1f, step_mod=%d, idle_exit_sec=%s",
        str(models_dir),
        poll_sec,
        stable_sec,
        step_mod,
        str(idle_exit_sec),
    )
    logging.info("已加载历史已评测ckpt数: %d", len(tested))

    while True:
        ckpts = discover_ckpts(models_dir, step_mod=step_mod)
        candidates = []
        for p in ckpts:
            sp = str(p.resolve())
            if sp in tested:
                continue
            if is_file_stable(p, last_sizes, stable_sec=stable_sec):
                candidates.append(p)

        if candidates:
            logging.info("发现 %d 个新模型待评测: %s", len(candidates), " -> ".join([c.stem for c in candidates]))
            for p in candidates:
                sp = str(p.resolve())
                t0 = time.perf_counter()
                model_tag, total, avg_acc, total_t, avg_t, avg_tr = evaluate_model(
                    model_path=sp,
                    outdir_root=outdir_root,
                    evaluator_pool=evaluator_pool,
                    dirs=dirs,
                )
                append_master(master_path, model_tag, total, avg_acc, total_t, avg_t, avg_tr)
                tested.add(sp)
                append_tested(tested_log, sp)
                logging.info("[%s] Done in %.2fs", model_tag, time.perf_counter() - t0)
            last_new_ts = time.time()
        else:
            if idle_exit_sec is not None and (time.time() - last_new_ts) > idle_exit_sec:
                logging.info("超过 %.0fs 没有新checkpoint，退出。", idle_exit_sec)
                break
            time.sleep(poll_sec)

    return master_path


def evaluate_once(
    model_path: str,
    outdir_root: Path,
    evaluator_pool: PersistentEvaluatorPool,
    dirs: List[Path],
) -> Path:
    master_path = outdir_root / "all_models_summary.tsv"
    ensure_master_header(master_path)
    t0 = time.perf_counter()
    model_tag, total, avg_acc, total_t, avg_t, avg_tr = evaluate_model(
        model_path=model_path,
        outdir_root=outdir_root,
        evaluator_pool=evaluator_pool,
        dirs=dirs,
    )
    append_master(master_path, model_tag, total, avg_acc, total_t, avg_t, avg_tr)
    logging.info("[%s] Done in %.2fs", model_tag, time.perf_counter() - t0)
    return master_path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=None, help="单个模型路径（与 --models_dir 互斥）")
    ap.add_argument("--models_dir", type=str, default=None, help="checkpoint目录，开启在线轮询评测")
    ap.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    ap.add_argument("--poll_sec", type=float, default=30.0, help="轮询间隔秒")
    ap.add_argument("--stable_sec", type=float, default=10.0, help="checkpoint最小稳定时长秒")
    ap.add_argument("--idle_exit_sec", type=float, default=None, help="超过该秒数无新ckpt则退出；默认一直运行")
    ap.add_argument("--step_mod", type=int, default=1, help="仅评测step%%step_mod==0的checkpoint；1表示不过滤")
    ap.add_argument("--clf_n_estimators", type=int, default=32, help="TabICLClassifier n_estimators")
    ap.add_argument(
        "--clf_batch_size",
        type=int,
        default=8,
        help="TabICLClassifier batch_size；设为-1表示None（单次处理所有ensemble）",
    )
    ap.add_argument("--clf_n_jobs", type=int, default=1, help="TabICLClassifier n_jobs（推荐1避免8进程CPU争抢）")
    ap.add_argument("--cpu_threads", type=int, default=1, help="每个GPU进程的CPU线程上限")
    ap.add_argument("--use_torch_compile", action="store_true", help="启用 torch.compile（默认关闭）")
    ap.add_argument(
        "--torch_compile_mode",
        type=str,
        default="reduce-overhead",
        help="torch.compile mode，设为none表示使用PyTorch默认",
    )
    ap.add_argument("--torch_compile_backend", type=str, default=None, help="torch.compile backend，默认PyTorch自动")
    ap.add_argument("--torch_compile_fullgraph", action="store_true", help="torch.compile fullgraph=True")
    ap.add_argument("--torch_compile_dynamic", action="store_true", help="torch.compile dynamic=True")
    ap.add_argument(
        "--torchinductor_cache_dir",
        type=str,
        default=None,
        help="TORCHINDUCTOR_CACHE_DIR，默认 <outdir>/_torchinductor_cache",
    )
    ap.add_argument(
        "--cache_root",
        type=str,
        default=None,
        help="预处理数据缓存目录，默认 <outdir>/_dataset_cache",
    )
    ap.add_argument(
        "--tested_log",
        type=str,
        default=None,
        help="已评测checkpoint记录文件，默认 <outdir>/tested_ckpts.txt",
    )
    return ap.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    data_root = Path(args.data_root)
    outdir_root = Path(args.outdir)
    outdir_root.mkdir(parents=True, exist_ok=True)

    tested_log = Path(args.tested_log) if args.tested_log else (outdir_root / "tested_ckpts.txt")
    clf_batch_size = None if int(args.clf_batch_size) <= 0 else int(args.clf_batch_size)
    clf_n_jobs = None if int(args.clf_n_jobs) <= 0 else int(args.clf_n_jobs)
    cpu_threads = max(1, int(args.cpu_threads))
    cache_root = Path(args.cache_root) if args.cache_root else (outdir_root / "_dataset_cache")
    use_torch_compile = bool(args.use_torch_compile)
    torch_compile_mode = args.torch_compile_mode
    if torch_compile_mode is not None and torch_compile_mode.strip().lower() in {"", "none"}:
        torch_compile_mode = None
    torch_compile_backend = args.torch_compile_backend
    if torch_compile_backend is not None and torch_compile_backend.strip().lower() in {"", "none"}:
        torch_compile_backend = None
    torch_compile_fullgraph = bool(args.torch_compile_fullgraph)
    torch_compile_dynamic = True if args.torch_compile_dynamic else None
    torchinductor_cache_dir = (
        str(Path(args.torchinductor_cache_dir))
        if args.torchinductor_cache_dir
        else str(outdir_root / "_torchinductor_cache")
    )

    dirs = [d for d in sorted(data_root.iterdir()) if d.is_dir()]
    summarize_task_types(dirs)

    gpu_devices = resolve_gpu_devices(FIXED_GPUS)
    if not gpu_devices:
        raise RuntimeError("未检测到可用GPU设备，无法执行并行评测。")
    logging.info("常驻评测池使用 %d 张GPU: %s", len(gpu_devices), ",".join(gpu_devices))
    if use_torch_compile:
        logging.info(
            "torch.compile 已开启: mode=%s, backend=%s, fullgraph=%s, dynamic=%s, cache=%s",
            str(torch_compile_mode),
            str(torch_compile_backend),
            str(torch_compile_fullgraph),
            str(torch_compile_dynamic),
            torchinductor_cache_dir,
        )

    evaluator_pool = PersistentEvaluatorPool(
        gpu_devices=gpu_devices,
        cache_root=cache_root,
        clf_n_estimators=int(args.clf_n_estimators),
        clf_batch_size=clf_batch_size,
        clf_n_jobs=clf_n_jobs,
        cpu_threads=cpu_threads,
        use_torch_compile=use_torch_compile,
        torch_compile_mode=torch_compile_mode,
        torch_compile_backend=torch_compile_backend,
        torch_compile_fullgraph=torch_compile_fullgraph,
        torch_compile_dynamic=torch_compile_dynamic,
        torchinductor_cache_dir=torchinductor_cache_dir,
    )

    try:
        if args.models_dir:
            models_dir = Path(args.models_dir)
            if not models_dir.exists():
                raise FileNotFoundError(f"--models_dir 不存在: {models_dir}")
            master_path = loop_eval_new_ckpts(
                models_dir=models_dir,
                outdir_root=outdir_root,
                poll_sec=float(args.poll_sec),
                stable_sec=float(args.stable_sec),
                idle_exit_sec=None if args.idle_exit_sec is None else float(args.idle_exit_sec),
                step_mod=int(args.step_mod),
                tested_log=tested_log,
                evaluator_pool=evaluator_pool,
                dirs=dirs,
            )
            print("\n汇总总表：", master_path)
            return

        model_path = args.model_path or DEFAULT_MODEL_PATH
        master_path = evaluate_once(
            model_path=model_path,
            outdir_root=outdir_root,
            evaluator_pool=evaluator_pool,
            dirs=dirs,
        )
        print("\n汇总总表：", master_path)
    finally:
        evaluator_pool.close()


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# """
# TALENT batch evaluation with online checkpoint watching.

# Key behavior:
# - If `--models_dir` is provided, keep polling this directory.
# - Whenever a new checkpoint appears and becomes stable, evaluate it once.
# - Each model evaluation shards datasets across up to 8 GPUs in parallel.
# - Write per-model outputs:
#   - <outdir>/<model_tag>/talent_detailed.txt
#   - <outdir>/<model_tag>/talent_summary.txt
# - Append global summary:
#   - <outdir>/all_models_summary.tsv
# """

# from __future__ import annotations

# import argparse
# import hashlib
# import inspect
# import json
# import logging
# import multiprocessing as mp
# import os
# import re
# import sys
# import time
# import warnings
# from pathlib import Path
# from typing import List, Optional, Tuple, Union

# import numpy as np
# import pandas as pd

# warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# # Allow running the script directly from repo root without editable install.
# REPO_ROOT = Path(__file__).resolve().parents[1]
# SRC_ROOT = REPO_ROOT / "src"
# if SRC_ROOT.exists():
#     sys.path.insert(0, str(SRC_ROOT))

# # ===== Fixed defaults =====
# DEFAULT_MODEL_PATH = "/vast/users/guangyi.chen/causal_group/zijian.li/LDM/Orion-MSP/stage1/checkpoint/dir2/step-27250d.ckpt"
# DEFAULT_DATA_ROOT = "/vast/users/guangyi.chen/causal_group/zijian.li/LDM/datasets"
# DEFAULT_OUTDIR = "evaluation_results_fulltrain_stage2"
# FIXED_GPUS = 8
# COERCE_NUMERIC = True
# MERGE_VAL = True
# SKIP_REGRESSION = True


# def convert_features(X: np.ndarray, enabled: bool) -> np.ndarray:
#     X = np.asarray(X)
#     if not enabled:
#         return X
#     if X.ndim == 1:
#         X = X.reshape(-1, 1)
#     df = pd.DataFrame(X)
#     encoded = pd.DataFrame(index=df.index)
#     for col in df.columns:
#         series = df.iloc[:, col]
#         numeric_series = pd.to_numeric(series, errors="coerce")
#         if series.isna().equals(numeric_series.isna()):
#             encoded[col] = numeric_series
#         else:
#             string_series = series.astype("string")
#             codes, uniques = pd.factorize(string_series, sort=True)
#             codes = codes.astype(np.int32)
#             if (codes == -1).any():
#                 codes[codes == -1] = len(uniques)
#             encoded[col] = codes
#     return encoded.fillna(0).values.astype(np.float32)


# def handle_missing_entries(X: np.ndarray, y: np.ndarray, *, context: str) -> tuple[np.ndarray, np.ndarray]:
#     X = np.asarray(X)
#     y = np.asarray(y)
#     df = pd.DataFrame(X)
#     y_series = pd.Series(y, index=df.index)
#     drop_mask = pd.Series(False, index=df.index)
#     for col in df.columns:
#         series = df.iloc[:, col]
#         numeric_series = pd.to_numeric(series, errors="coerce")
#         if series.isna().equals(numeric_series.isna()):
#             nan_mask = numeric_series.isna()
#             if nan_mask.any():
#                 mean_value = float(numeric_series.mean(skipna=True))
#                 if np.isnan(mean_value):
#                     mean_value = 0.0
#                 df.iloc[:, col] = numeric_series.fillna(mean_value)
#         else:
#             nan_mask = series.isna()
#             if nan_mask.any():
#                 drop_mask |= nan_mask

#     if drop_mask.any():
#         drop_count = int(drop_mask.sum())
#         df = df.loc[~drop_mask].copy()
#         y_series = y_series.loc[df.index]
#         logging.info("%s: 删除 %d 行包含字符串缺失值", context, drop_count)
#     return df.values, y_series.values


# def count_missing(values: np.ndarray) -> int:
#     if values is None:
#         return 0
#     arr = np.asarray(values)
#     if arr.dtype.kind in {"f", "c"}:
#         return int(np.isnan(arr).sum())
#     mask = pd.isna(pd.DataFrame(arr))
#     return int(mask.values.sum())


# def log_nan_presence(
#     context: str,
#     values: np.ndarray,
#     *,
#     dataset_id: str | None = None,
#     missing_registry: set[str] | None = None,
# ) -> None:
#     missing = count_missing(values)
#     if missing:
#         logging.warning("%s: 原始数据包含 %d 个 NaN/缺失值", context, missing)
#         if dataset_id and missing_registry is not None:
#             missing_registry.add(dataset_id)


# def load_array(file_path: Path) -> np.ndarray:
#     suffix = file_path.suffix.lower()
#     if suffix in {".npy", ".npz"}:
#         try:
#             arr = np.load(file_path, allow_pickle=False)
#         except ValueError:
#             arr = np.load(file_path, allow_pickle=True)
#         if isinstance(arr, np.lib.npyio.NpzFile):
#             arr = arr[list(arr.files)[0]]
#         return np.asarray(arr)
#     if suffix == ".parquet":
#         return pd.read_parquet(file_path).values
#     sep = "\t" if suffix == ".tsv" else None
#     return pd.read_csv(file_path, sep=sep, header=None).values


# def find_data_files(dataset_dir: Path):
#     files = [p for p in dataset_dir.iterdir() if p.is_file()]
#     lower = {p.name.lower(): p for p in files}

#     def by_suffix(key: str):
#         for name, p in lower.items():
#             if name.endswith(key):
#                 return p
#         return None

#     n_train = by_suffix("n_train.npy")
#     c_train = by_suffix("c_train.npy")
#     y_train = by_suffix("y_train.npy")
#     n_val = by_suffix("n_val.npy")
#     c_val = by_suffix("c_val.npy")
#     y_val = by_suffix("y_val.npy")
#     n_test = by_suffix("n_test.npy")
#     c_test = by_suffix("c_test.npy")
#     y_test = by_suffix("y_test.npy")

#     if y_train and y_test and (n_train or c_train) and (n_test or c_test):
#         val_pair = None
#         if y_val and (n_val or c_val):
#             val_pair = (n_val, c_val, y_val)
#         return (n_train, c_train, y_train), val_pair, (n_test, c_test, y_test)

#     table_candidates = [p for p in files if p.suffix.lower() in {".npy", ".npz", ".csv", ".tsv", ".parquet"}]
#     if len(table_candidates) == 1:
#         return table_candidates[0], None, None
#     return None, None, None


# def load_pair(
#     X_path: Path,
#     y_path: Path,
#     context: str = "",
#     coerce_numeric: bool = False,
#     dataset_id: str | None = None,
#     missing_registry: set[str] | None = None,
# ):
#     X = load_array(X_path)
#     y = load_array(y_path)
#     log_nan_presence(f"{context or X_path.stem}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
#     log_nan_presence(f"{context or X_path.stem}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)
#     X = np.asarray(X)
#     y = np.asarray(y)
#     if X.ndim == 1:
#         X = X.reshape(-1, 1)
#     if y.ndim > 1:
#         if y.shape[1] == 1:
#             y = y.squeeze(1)
#         elif y.shape[0] == 1:
#             y = y.squeeze(0)
#     X, y = handle_missing_entries(X, y, context=context or X_path.stem)
#     X = convert_features(X, coerce_numeric)
#     return X, y


# def load_split(
#     num_path: Optional[Path],
#     cat_path: Optional[Path],
#     y_path: Path,
#     context: str = "",
#     coerce_numeric: bool = False,
#     dataset_id: str | None = None,
#     missing_registry: set[str] | None = None,
# ):
#     feats = []
#     base = context or (num_path.stem if num_path else (cat_path.stem if cat_path else y_path.stem))

#     if num_path:
#         Xn = np.asarray(load_array(num_path))
#         if Xn.ndim == 1:
#             Xn = Xn.reshape(-1, 1)
#         log_nan_presence(f"{base}-num_raw", Xn, dataset_id=dataset_id, missing_registry=missing_registry)
#         feats.append(Xn)
#     if cat_path:
#         Xc = np.asarray(load_array(cat_path))
#         if Xc.ndim == 1:
#             Xc = Xc.reshape(-1, 1)
#         log_nan_presence(f"{base}-cat_raw", Xc, dataset_id=dataset_id, missing_registry=missing_registry)
#         feats.append(Xc)
#     if not feats:
#         raise ValueError("缺少数值/类别特征文件")

#     n = feats[0].shape[0]
#     for i, f in enumerate(feats):
#         if f.shape[0] != n:
#             raise ValueError(f"特征数量不一致: #{i} 有 {f.shape[0]} vs {n}")

#     X = feats[0] if len(feats) == 1 else np.concatenate(feats, axis=1)
#     log_nan_presence(f"{base}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
#     y = np.asarray(load_array(y_path))
#     log_nan_presence(f"{base}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)
#     if y.ndim > 1:
#         if y.shape[1] == 1:
#             y = y.squeeze(1)
#         elif y.shape[0] == 1:
#             y = y.squeeze(0)
#     X, y = handle_missing_entries(X, y, context=base)
#     X = convert_features(X, coerce_numeric)
#     return X, y


# def load_table(
#     file_path: Union[Path, Tuple],
#     context: str = "",
#     coerce_numeric: bool = False,
#     dataset_id: str | None = None,
#     missing_registry: set[str] | None = None,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     if isinstance(file_path, (tuple, list)):
#         if len(file_path) == 2:
#             Xp, yp = Path(file_path[0]), Path(file_path[1])
#             return load_pair(
#                 Xp,
#                 yp,
#                 context=context,
#                 coerce_numeric=coerce_numeric,
#                 dataset_id=dataset_id,
#                 missing_registry=missing_registry,
#             )
#         if len(file_path) == 3:
#             num_path, cat_path, y_path = file_path
#             return load_split(
#                 Path(num_path) if num_path else None,
#                 Path(cat_path) if cat_path else None,
#                 Path(y_path),
#                 context=context,
#                 coerce_numeric=coerce_numeric,
#                 dataset_id=dataset_id,
#                 missing_registry=missing_registry,
#             )
#         raise ValueError(f"Unsupported tuple for load_table: {file_path}")

#     path: Path = Path(file_path)
#     suffix = path.suffix.lower()
#     if suffix in {".npy", ".npz"}:
#         try:
#             arr = np.load(path, allow_pickle=False)
#         except ValueError:
#             arr = np.load(path, allow_pickle=True)
#         if isinstance(arr, np.lib.npyio.NpzFile):
#             arr = arr[list(arr.files)[0]]
#         data = np.asarray(arr)
#     elif suffix == ".parquet":
#         data = pd.read_parquet(path).values
#     else:
#         sep = "\t" if suffix == ".tsv" else None
#         data = pd.read_csv(path, sep=sep, header=None).values

#     if data.ndim == 1:
#         raise ValueError(f"Unsupported 1D data in {path}")

#     log_target = context or str(path)
#     log_nan_presence(f"{log_target}-raw", data, dataset_id=dataset_id, missing_registry=missing_registry)

#     col0 = data[:, 0]
#     try:
#         uniques0 = np.unique(col0)
#     except Exception:
#         uniques0 = np.array([])

#     if 0 < uniques0.size < max(2, data.shape[0] // 2):
#         y = data[:, 0]
#         X = data[:, 1:]
#         which = "first"
#     else:
#         y = data[:, -1]
#         X = data[:, :-1]
#         which = "last"

#     log_nan_presence(f"{log_target}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
#     log_nan_presence(f"{log_target}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)

#     X = np.asarray(pd.DataFrame(X).values)
#     y = pd.Series(y).values
#     X, y = handle_missing_entries(X, y, context=log_target)
#     X = convert_features(X, coerce_numeric)
#     logging.info("%s: 使用单文件启发式拆分标签 (取 %s 列)", log_target, which)
#     return X, y


# def load_dataset_info(dataset_dir: Path) -> Optional[dict]:
#     p = dataset_dir / "info.json"
#     if not p.exists():
#         return None
#     try:
#         with open(p, "r", encoding="utf-8") as f:
#             return json.load(f)
#     except Exception as exc:
#         logging.warning("读取 %s 失败: %s", p, exc)
#         return None


# def summarize_task_types(dirs: List[Path]) -> None:
#     counts = {"regression": 0, "binclass": 0, "multiclass": 0, "unknown": 0}
#     for d in dirs:
#         info = load_dataset_info(d)
#         t = str(info.get("task_type", "")).lower() if info else ""
#         if not t:
#             counts["unknown"] += 1
#         elif t in counts:
#             counts["regression" if t == "regression" else t] += 1
#         else:
#             counts["unknown"] += 1
#     logging.info(
#         "任务统计: regression=%d, binclass=%d, multiclass=%d, unknown=%d, 总计=%d",
#         counts["regression"],
#         counts["binclass"],
#         counts["multiclass"],
#         counts["unknown"],
#         len(dirs),
#     )


# def resolve_gpu_devices(max_gpus: int = FIXED_GPUS) -> List[str]:
#     env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
#     if env:
#         devices = [x.strip() for x in env.split(",") if x.strip()]
#     else:
#         try:
#             import torch

#             n = torch.cuda.device_count()
#         except Exception:
#             n = 0
#         devices = [str(i) for i in range(n)] if n > 0 else [str(i) for i in range(max_gpus)]

#     if len(devices) > max_gpus:
#         devices = devices[:max_gpus]
#     return devices


# def build_classifier(
#     model_path: str,
#     n_estimators: int,
#     batch_size: Optional[int],
#     n_jobs: Optional[int],
#     use_torch_compile: bool,
#     torch_compile_mode: Optional[str],
#     torch_compile_backend: Optional[str],
#     torch_compile_fullgraph: bool,
#     torch_compile_dynamic: Optional[bool],
# ):
#     """Build classifier with backward-compatible kwargs across tabicl versions."""
#     from tabicl.sklearn.classifier import TabICLClassifier

#     kwargs = {
#         "verbose": False,
#         "model_path": model_path,
#         "device": "cuda",
#         "use_amp": True,
#         "n_estimators": n_estimators,
#     }
#     if batch_size is not None:
#         kwargs["batch_size"] = batch_size
#     if n_jobs is not None:
#         kwargs["n_jobs"] = n_jobs
#     try:
#         sig = inspect.signature(TabICLClassifier.__init__)
#         params = sig.parameters
#         if "use_kv_cache" in params:
#             kwargs["use_kv_cache"] = True
#         if use_torch_compile and "use_torch_compile" in params:
#             kwargs["use_torch_compile"] = True
#             if "torch_compile_mode" in params:
#                 kwargs["torch_compile_mode"] = torch_compile_mode
#             if "torch_compile_backend" in params:
#                 kwargs["torch_compile_backend"] = torch_compile_backend
#             if "torch_compile_fullgraph" in params:
#                 kwargs["torch_compile_fullgraph"] = torch_compile_fullgraph
#             if "torch_compile_dynamic" in params:
#                 kwargs["torch_compile_dynamic"] = torch_compile_dynamic
#     except Exception:
#         pass
#     return TabICLClassifier(**kwargs)


# def preload_model_once(clf, gpu_device: str) -> None:
#     """Force one-time checkpoint load and prevent per-dataset reload in legacy classifier."""
#     load_fn = getattr(clf, "_load_model", None)
#     if not callable(load_fn):
#         return

#     t0 = time.perf_counter()
#     load_fn()

#     # Newer classifier variants use _loaded_model_key in _ensure_model_loaded.
#     if hasattr(clf, "_get_model_load_key"):
#         try:
#             clf._loaded_model_key = clf._get_model_load_key()
#         except Exception:
#             pass

#     # Legacy classifier calls _load_model() in every fit(); make it a no-op after preload.
#     def _skip_reloading():
#         return None

#     try:
#         clf._load_model = _skip_reloading
#     except Exception:
#         pass

#     logging.info("[GPU %s] 模型预加载完成: %.2fs", gpu_device, time.perf_counter() - t0)


# def _set_cpu_thread_limits(cpu_threads: int) -> None:
#     cpu_threads = max(1, int(cpu_threads))
#     os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
#     os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
#     os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
#     os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_threads)
#     os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_threads)

#     try:
#         from threadpoolctl import threadpool_limits

#         threadpool_limits(limits=cpu_threads)
#     except Exception:
#         pass


# def _dataset_cache_file(dataset_dir: Path, cache_root: Path) -> Path:
#     dataset_key = hashlib.md5(str(dataset_dir.resolve()).encode("utf-8")).hexdigest()[:12]
#     return cache_root / f"{dataset_dir.name}__{dataset_key}.npz"


# def _load_cached_dataset(cache_file: Path):
#     try:
#         with np.load(cache_file, allow_pickle=True) as z:
#             X_train = z["X_train"]
#             y_train = z["y_train"]
#             X_test = z["X_test"]
#             y_test = z["y_test"]
#             train_ratio = float(z["train_ratio"])
#         return X_train, y_train, X_test, y_test, train_ratio
#     except Exception:
#         return None


# def _prepare_dataset_payload(dataset_dir: Path, cache_root: Path):
#     info = load_dataset_info(dataset_dir)
#     ttype = str(info.get("task_type", "")).lower() if info else None
#     if SKIP_REGRESSION and ttype == "regression":
#         return None, "回归任务", False

#     cache_file = _dataset_cache_file(dataset_dir, cache_root)
#     if cache_file.exists():
#         cached = _load_cached_dataset(cache_file)
#         if cached is not None:
#             X_train, y_train, X_test, y_test, train_ratio = cached
#             return (
#                 {
#                     "X_train": X_train,
#                     "y_train": y_train,
#                     "X_test": X_test,
#                     "y_test": y_test,
#                     "train_ratio": train_ratio,
#                 },
#                 None,
#                 True,
#             )
#         logging.warning("%s: 缓存文件损坏，重新构建: %s", dataset_dir.name, cache_file)

#     missing_datasets: set[str] = set()
#     train_path, val_path, test_path = find_data_files(dataset_dir)
#     if train_path is None and test_path is None:
#         return None, "未识别数据文件", False
#     if not (train_path and test_path):
#         return None, "只有单文件，当前策略跳过", False

#     X_train, y_train = load_table(
#         train_path,
#         context=f"{dataset_dir.name}-train",
#         coerce_numeric=COERCE_NUMERIC,
#         dataset_id=dataset_dir.name,
#         missing_registry=missing_datasets,
#     )
#     X_test, y_test = load_table(
#         test_path,
#         context=f"{dataset_dir.name}-test",
#         coerce_numeric=COERCE_NUMERIC,
#         dataset_id=dataset_dir.name,
#         missing_registry=missing_datasets,
#     )

#     if dataset_dir.name in missing_datasets:
#         logging.info("%s: 原始数据包含缺失值", dataset_dir.name)

#     if val_path is not None and MERGE_VAL:
#         try:
#             X_val, y_val = load_table(
#                 val_path,
#                 context=f"{dataset_dir.name}-val",
#                 coerce_numeric=COERCE_NUMERIC,
#                 dataset_id=dataset_dir.name,
#                 missing_registry=missing_datasets,
#             )
#             if X_val.ndim == 3 and X_val.shape[1] == 1:
#                 X_val = X_val.squeeze(1)
#             X_val = X_val.astype(np.float32, copy=False)
#             X_train = np.concatenate([X_train, X_val], axis=0)
#             y_train = np.concatenate([y_train, y_val], axis=0)
#             logging.info("%s: 已将 val(%d) 并入 train", dataset_dir.name, int(X_val.shape[0]))
#         except Exception as exc:
#             logging.warning("%s: 合并 val 失败：%s", dataset_dir.name, exc)

#     if X_train.ndim == 3 and X_train.shape[1] == 1:
#         X_train = X_train.squeeze(1)
#     if X_test.ndim == 3 and X_test.shape[1] == 1:
#         X_test = X_test.squeeze(1)
#     X_train = X_train.astype(np.float32, copy=False)
#     X_test = X_test.astype(np.float32, copy=False)

#     train_count = int(X_train.shape[0])
#     test_count = int(X_test.shape[0])
#     total_count = train_count + test_count
#     train_ratio = float(train_count / total_count) if total_count > 0 else float("nan")

#     cache_root.mkdir(parents=True, exist_ok=True)
#     try:
#         np.savez(
#             cache_file,
#             X_train=X_train,
#             y_train=y_train,
#             X_test=X_test,
#             y_test=y_test,
#             train_ratio=np.array(train_ratio, dtype=np.float32),
#         )
#     except Exception as exc:
#         logging.warning("%s: 写缓存失败 (%s): %s", dataset_dir.name, cache_file, exc)

#     return (
#         {
#             "X_train": X_train,
#             "y_train": y_train,
#             "X_test": X_test,
#             "y_test": y_test,
#             "train_ratio": train_ratio,
#         },
#         None,
#         False,
#     )


# def _worker_loop(
#     task_queue,
#     result_queue,
#     gpu_device: str,
#     cache_root: str,
#     clf_n_estimators: int,
#     clf_batch_size: Optional[int],
#     clf_n_jobs: Optional[int],
#     cpu_threads: int,
#     use_torch_compile: bool,
#     torch_compile_mode: Optional[str],
#     torch_compile_backend: Optional[str],
#     torch_compile_fullgraph: bool,
#     torch_compile_dynamic: Optional[bool],
#     torchinductor_cache_dir: Optional[str],
# ):
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
#     if torchinductor_cache_dir:
#         os.environ["TORCHINDUCTOR_CACHE_DIR"] = torchinductor_cache_dir
#     _set_cpu_thread_limits(cpu_threads)
#     try:
#         import torch

#         torch.cuda.set_device(0)
#         torch.set_num_threads(max(1, int(cpu_threads)))
#     except Exception:
#         pass

#     logging.info("[GPU %s] 启动，常驻worker模式", gpu_device)

#     clf = None
#     current_model_path = None
#     cache_root_path = Path(cache_root)

#     while True:
#         task = task_queue.get()
#         if task is None:
#             break

#         eval_id = int(task["eval_id"])
#         model_path = str(task["model_path"])
#         d = Path(task["dataset_dir"])

#         if clf is None or current_model_path != model_path:
#             clf = build_classifier(
#                 model_path=model_path,
#                 n_estimators=clf_n_estimators,
#                 batch_size=clf_batch_size,
#                 n_jobs=clf_n_jobs,
#                 use_torch_compile=use_torch_compile,
#                 torch_compile_mode=torch_compile_mode,
#                 torch_compile_backend=torch_compile_backend,
#                 torch_compile_fullgraph=torch_compile_fullgraph,
#                 torch_compile_dynamic=torch_compile_dynamic,
#             )
#             preload_model_once(clf, gpu_device)
#             current_model_path = model_path

#         ds_t0 = time.perf_counter()
#         try:
#             payload, skip_reason, cache_hit = _prepare_dataset_payload(d, cache_root_path)
#             if payload is None:
#                 result_queue.put(
#                     {
#                         "eval_id": eval_id,
#                         "status": "skip",
#                         "dataset": d.name,
#                         "reason": skip_reason,
#                     }
#                 )
#                 continue

#             X_train = payload["X_train"]
#             y_train = payload["y_train"]
#             X_test = payload["X_test"]
#             y_test = payload["y_test"]
#             train_ratio = float(payload["train_ratio"])

#             prep_time = time.perf_counter() - ds_t0

#             t_fit = time.perf_counter()
#             clf.fit(X_train, y_train)
#             fit_time = time.perf_counter() - t_fit

#             t_pred = time.perf_counter()
#             y_pred = clf.predict(X_test)
#             pred_time = time.perf_counter() - t_pred

#             acc = float(np.mean(y_pred == y_test))
#             infer_time = fit_time + pred_time
#             total_e2e = time.perf_counter() - ds_t0
#             logging.info(
#                 "[GPU %s] %s: acc=%.4f, infer=%.2fs (fit=%.2fs, pred=%.2fs), prep=%.2fs, e2e=%.2fs, train_ratio=%.4f",
#                 gpu_device,
#                 d.name,
#                 acc,
#                 infer_time,
#                 fit_time,
#                 pred_time,
#                 prep_time,
#                 total_e2e,
#                 train_ratio,
#             )
#             result_queue.put(
#                 {
#                     "eval_id": eval_id,
#                     "status": "ok",
#                     "dataset": d.name,
#                     "acc": acc,
#                     "infer_time": infer_time,
#                     "train_ratio": train_ratio,
#                     "prep_time": prep_time,
#                     "fit_time": fit_time,
#                     "pred_time": pred_time,
#                     "total_e2e": total_e2e,
#                     "cache_hit": cache_hit,
#                 }
#             )

#         except Exception as exc:
#             logging.exception("[GPU %s] 评测失败 %s: %s", gpu_device, d.name, exc)
#             result_queue.put(
#                 {
#                     "eval_id": eval_id,
#                     "status": "error",
#                     "dataset": d.name,
#                     "error": str(exc),
#                 }
#             )


# class PersistentEvaluatorPool:
#     def __init__(
#         self,
#         gpu_devices: List[str],
#         cache_root: Path,
#         clf_n_estimators: int,
#         clf_batch_size: Optional[int],
#         clf_n_jobs: Optional[int],
#         cpu_threads: int,
#         use_torch_compile: bool,
#         torch_compile_mode: Optional[str],
#         torch_compile_backend: Optional[str],
#         torch_compile_fullgraph: bool,
#         torch_compile_dynamic: Optional[bool],
#         torchinductor_cache_dir: Optional[str],
#     ) -> None:
#         if not gpu_devices:
#             raise RuntimeError("未检测到可用GPU设备，无法执行并行评测。")

#         self.ctx = mp.get_context("spawn")
#         self.task_queue = self.ctx.Queue(maxsize=max(64, len(gpu_devices) * 8))
#         self.result_queue = self.ctx.Queue()
#         self.processes = []
#         self.eval_id = 0
#         self.cache_root = cache_root
#         self.cache_root.mkdir(parents=True, exist_ok=True)

#         for gpu_dev in gpu_devices:
#             p = self.ctx.Process(
#                 target=_worker_loop,
#                 args=(
#                     self.task_queue,
#                     self.result_queue,
#                     gpu_dev,
#                     str(self.cache_root),
#                     clf_n_estimators,
#                     clf_batch_size,
#                     clf_n_jobs,
#                     cpu_threads,
#                     use_torch_compile,
#                     torch_compile_mode,
#                     torch_compile_backend,
#                     torch_compile_fullgraph,
#                     torch_compile_dynamic,
#                     torchinductor_cache_dir,
#                 ),
#                 daemon=False,
#             )
#             p.start()
#             self.processes.append(p)

#         logging.info("常驻评测池已启动: %d workers, cache=%s", len(self.processes), self.cache_root)

#     def close(self) -> None:
#         for _ in self.processes:
#             self.task_queue.put(None)
#         for p in self.processes:
#             p.join(timeout=5)
#             if p.is_alive():
#                 p.terminate()
#                 p.join(timeout=2)

#     def evaluate_model(self, model_path: str, dirs: List[Path]):
#         eval_id = self.eval_id
#         self.eval_id += 1
#         total_tasks = len(dirs)

#         for d in dirs:
#             self.task_queue.put({"eval_id": eval_id, "model_path": model_path, "dataset_dir": str(d)})

#         results = []
#         done = 0
#         skip_count = 0
#         err_count = 0
#         cache_hits = 0
#         while done < total_tasks:
#             msg = self.result_queue.get()
#             if int(msg.get("eval_id", -1)) != eval_id:
#                 continue
#             done += 1
#             status = msg.get("status")
#             if status == "ok":
#                 results.append(
#                     (
#                         msg["dataset"],
#                         float(msg["acc"]),
#                         float(msg["infer_time"]),
#                         float(msg["train_ratio"]),
#                         float(msg["prep_time"]),
#                         float(msg["fit_time"]),
#                         float(msg["pred_time"]),
#                         float(msg["total_e2e"]),
#                     )
#                 )
#                 if msg.get("cache_hit"):
#                     cache_hits += 1
#             elif status == "skip":
#                 skip_count += 1
#             else:
#                 err_count += 1
#                 logging.warning("数据集失败: %s (%s)", msg.get("dataset"), msg.get("error"))

#         logging.info(
#             "模型 %s 评测完成: ok=%d, skip=%d, error=%d, cache_hit=%d/%d",
#             Path(model_path).stem,
#             len(results),
#             skip_count,
#             err_count,
#             cache_hits,
#             total_tasks,
#         )
#         return results


# def evaluate_model(
#     model_path: str,
#     outdir_root: Path,
#     evaluator_pool: PersistentEvaluatorPool,
#     dirs: List[Path],
# ) -> Tuple[str, int, float, float, float, float]:
#     """
#     Returns:
#     (model_tag, total_datasets, avg_acc, total_time, avg_time, avg_train_ratio)
#     """
#     model_tag = Path(model_path).stem
#     outdir = outdir_root / model_tag
#     outdir.mkdir(parents=True, exist_ok=True)

#     results = evaluator_pool.evaluate_model(model_path=model_path, dirs=dirs)
#     results.sort(key=lambda x: x[0])

#     detailed_path = outdir / "talent_detailed.txt"
#     summary_path = outdir / "talent_summary.txt"

#     if results:
#         with open(detailed_path, "w") as f:
#             f.write("dataset\taccuracy\ttime_s\ttrain_ratio\tprep_s\tfit_s\tpredict_s\ttotal_e2e_s\n")
#             for name, acc, dur, tr, prep_s, fit_s, pred_s, e2e_s in results:
#                 tr_str = f"{tr:.6f}" if tr == tr else "nan"
#                 f.write(f"{name}\t{acc:.6f}\t{dur:.3f}\t{tr_str}\t{prep_s:.3f}\t{fit_s:.3f}\t{pred_s:.3f}\t{e2e_s:.3f}\n")

#         total_time = sum(dur for _, _, dur, _, _, _, _, _ in results)
#         avg_time = total_time / len(results)
#         avg_acc = sum(acc for _, acc, _, _, _, _, _, _ in results) / len(results)
#         tr_values = [tr for _, _, _, tr, _, _, _, _ in results if tr == tr]
#         avg_prep_time = sum(prep for _, _, _, _, prep, _, _, _ in results) / len(results)
#         avg_fit_time = sum(fit for _, _, _, _, _, fit, _, _ in results) / len(results)
#         avg_pred_time = sum(pred for _, _, _, _, _, _, pred, _ in results) / len(results)
#         avg_e2e_time = sum(e2e for _, _, _, _, _, _, _, e2e in results) / len(results)
#         avg_train_ratio = (sum(tr_values) / len(tr_values)) if tr_values else float("nan")

#         with open(summary_path, "w") as f:
#             f.write(f"Model: {model_tag}\n")
#             f.write(f"Total datasets: {len(results)}\n")
#             f.write(f"Average accuracy: {avg_acc:.6f}\n")
#             f.write(f"Total time s: {total_time:.3f}\n")
#             f.write(f"Average time s: {avg_time:.3f}\n")
#             f.write(f"Average prep time s: {avg_prep_time:.3f}\n")
#             f.write(f"Average fit time s: {avg_fit_time:.3f}\n")
#             f.write(f"Average predict time s: {avg_pred_time:.3f}\n")
#             f.write(f"Average end-to-end time s: {avg_e2e_time:.3f}\n")
#             f.write(f"Average train_ratio: {avg_train_ratio:.6f}\n")

#         logging.info("[%s] 汇总完成：%s / %s", model_tag, detailed_path, summary_path)
#         return model_tag, len(results), avg_acc, total_time, avg_time, avg_train_ratio

#     logging.info("[%s] 没有成功的评测结果。", model_tag)
#     return model_tag, 0, float("nan"), 0.0, float("nan"), float("nan")


# def _extract_last_int(stem: str) -> Optional[int]:
#     nums = re.findall(r"\d+", stem)
#     if not nums:
#         return None
#     return int(nums[-1])


# def discover_ckpts(models_dir: Path, step_mod: int) -> List[Path]:
#     files = [p for p in models_dir.iterdir() if p.is_file() and p.suffix.lower() in {".ckpt", ".pt", ".pth"}]

#     def sort_key(p: Path):
#         step = _extract_last_int(p.stem)
#         if step is not None:
#             return (0, step, p.stem)
#         return (1, int(p.stat().st_mtime), p.stem)

#     ordered = sorted(files, key=sort_key)
#     if step_mod <= 1:
#         return ordered

#     filtered = []
#     for p in ordered:
#         step = _extract_last_int(p.stem)
#         if step is None or step % step_mod == 0:
#             filtered.append(p)
#     return filtered


# def is_file_stable(p: Path, last_sizes: dict[str, int], stable_sec: float) -> bool:
#     key = str(p)
#     try:
#         st = p.stat()
#     except Exception:
#         return False

#     age = time.time() - st.st_mtime
#     if age < stable_sec:
#         return False

#     size = st.st_size
#     prev = last_sizes.get(key)
#     last_sizes[key] = size
#     return prev is not None and prev == size


# def ensure_master_header(master_path: Path) -> None:
#     if not master_path.exists():
#         master_path.parent.mkdir(parents=True, exist_ok=True)
#         with open(master_path, "w") as f:
#             f.write("model_name\ttotal_datasets\taverage_accuracy\ttotal_time_s\taverage_time_s\taverage_train_ratio\n")


# def append_master(master_path: Path, model_tag: str, total: int, avg_acc: float, total_t: float, avg_t: float, avg_tr: float):
#     avg_acc_str = f"{avg_acc:.6f}" if avg_acc == avg_acc else "nan"
#     avg_t_str = f"{avg_t:.3f}" if avg_t == avg_t else "nan"
#     avg_tr_str = f"{avg_tr:.6f}" if avg_tr == avg_tr else "nan"
#     with open(master_path, "a") as f:
#         f.write(f"{model_tag}\t{total}\t{avg_acc_str}\t{total_t:.3f}\t{avg_t_str}\t{avg_tr_str}\n")


# def load_tested(tested_log: Path) -> set[str]:
#     if not tested_log.exists():
#         return set()
#     with open(tested_log, "r", encoding="utf-8") as f:
#         return {str(Path(line.strip()).resolve()) for line in f if line.strip()}


# def append_tested(tested_log: Path, ckpt_path: str) -> None:
#     tested_log.parent.mkdir(parents=True, exist_ok=True)
#     with open(tested_log, "a", encoding="utf-8") as f:
#         f.write(str(Path(ckpt_path).resolve()) + "\n")


# def _safe_float(value) -> float:
#     try:
#         return float(value)
#     except Exception:
#         return float("nan")


# def load_master_rows(master_path: Path) -> List[dict]:
#     if not master_path.exists():
#         return []
#     try:
#         df = pd.read_csv(master_path, sep="\t")
#     except Exception:
#         return []

#     rows = []
#     for _, row in df.iterrows():
#         rows.append(
#             {
#                 "model_name": str(row.get("model_name", "")),
#                 "total_datasets": int(_safe_float(row.get("total_datasets", 0))),
#                 "average_accuracy": _safe_float(row.get("average_accuracy", float("nan"))),
#                 "total_time_s": _safe_float(row.get("total_time_s", float("nan"))),
#                 "average_time_s": _safe_float(row.get("average_time_s", float("nan"))),
#                 "average_train_ratio": _safe_float(row.get("average_train_ratio", float("nan"))),
#             }
#         )
#     return rows


# def get_model_avg_acc_from_master(master_path: Path, model_tag: str) -> Optional[float]:
#     rows = load_master_rows(master_path)
#     found = [r for r in rows if r["model_name"] == model_tag and np.isfinite(r["average_accuracy"])]
#     if not found:
#         return None
#     return float(found[-1]["average_accuracy"])


# def write_best_result(master_path: Path, best_result_path: Path) -> Optional[dict]:
#     rows = load_master_rows(master_path)
#     valid = [r for r in rows if np.isfinite(r["average_accuracy"])]
#     if not valid:
#         return None
#     best = max(valid, key=lambda r: r["average_accuracy"])
#     best_result_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(best_result_path, "w", encoding="utf-8") as f:
#         f.write("model_name\taverage_accuracy\ttotal_datasets\taverage_time_s\ttotal_time_s\taverage_train_ratio\n")
#         f.write(
#             f"{best['model_name']}\t{best['average_accuracy']:.6f}\t{best['total_datasets']}"
#             f"\t{best['average_time_s']:.3f}\t{best['total_time_s']:.3f}\t{best['average_train_ratio']:.6f}\n"
#         )
#     return best


# def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
#     if not intervals:
#         return []
#     merged = []
#     for lo, hi in sorted(intervals):
#         if not merged or lo > merged[-1][1]:
#             merged.append([lo, hi])
#         else:
#             merged[-1][1] = max(merged[-1][1], hi)
#     return [(a, b) for a, b in merged]


# def select_fine_candidates_from_stable_regions(
#     coarse_results: List[dict],
#     all_ckpts: List[Path],
#     fine_step_mod: int,
#     stable_acc_delta: float,
# ) -> List[Path]:
#     # Keep checkpoints with numeric step and valid avg_acc
#     points = [
#         r
#         for r in coarse_results
#         if r.get("step") is not None and np.isfinite(r.get("avg_acc", float("nan")))
#     ]
#     points.sort(key=lambda r: int(r["step"]))
#     if len(points) < 2:
#         return []

#     intervals = []
#     for a, b in zip(points, points[1:]):
#         if abs(float(a["avg_acc"]) - float(b["avg_acc"])) <= stable_acc_delta:
#             lo, hi = sorted((int(a["step"]), int(b["step"])))
#             intervals.append((lo, hi))
#     merged = _merge_intervals(intervals)
#     if not merged:
#         return []

#     selected = []
#     for p in all_ckpts:
#         step = _extract_last_int(p.stem)
#         if step is None:
#             continue
#         if fine_step_mod > 1 and step % fine_step_mod != 0:
#             continue
#         if any(lo <= step <= hi for lo, hi in merged):
#             selected.append(p)
#     return selected


# def select_fine_candidates_from_acc_window(
#     coarse_results: List[dict],
#     all_ckpts: List[Path],
#     fine_step_mod: int,
#     fine_acc_min: float,
#     fine_acc_max: float,
# ) -> List[Path]:
#     # Keep checkpoints with numeric step and valid avg_acc
#     points = [
#         r
#         for r in coarse_results
#         if r.get("step") is not None and np.isfinite(r.get("avg_acc", float("nan")))
#     ]
#     points.sort(key=lambda r: int(r["step"]))
#     if len(points) < 2:
#         return []

#     acc_lo = float(min(fine_acc_min, fine_acc_max))
#     acc_hi = float(max(fine_acc_min, fine_acc_max))

#     intervals = []
#     for a, b in zip(points, points[1:]):
#         a_acc = float(a["avg_acc"])
#         b_acc = float(b["avg_acc"])
#         seg_lo, seg_hi = min(a_acc, b_acc), max(a_acc, b_acc)
#         # Segment intersects target accuracy window
#         if seg_hi < acc_lo or seg_lo > acc_hi:
#             continue
#         lo, hi = sorted((int(a["step"]), int(b["step"])))
#         intervals.append((lo, hi))

#     merged = _merge_intervals(intervals)
#     if not merged:
#         return []

#     selected = []
#     for p in all_ckpts:
#         step = _extract_last_int(p.stem)
#         if step is None:
#             continue
#         if fine_step_mod > 1 and step % fine_step_mod != 0:
#             continue
#         if any(lo <= step <= hi for lo, hi in merged):
#             selected.append(p)
#     return selected


# def evaluate_two_stage_ckpts(
#     models_dir: Path,
#     outdir_root: Path,
#     tested_log: Path,
#     evaluator_pool: PersistentEvaluatorPool,
#     dirs: List[Path],
#     coarse_step_mod: int,
#     fine_step_mod: int,
#     stable_acc_delta: float,
#     fine_acc_min: Optional[float],
#     fine_acc_max: Optional[float],
#     best_result_path: Path,
# ) -> Path:
#     """
#     Two-stage evaluation:
#     1) Coarse pass (e.g., every 100 steps)
#     2) Fine pass in selected regions:
#        - default: stable-accuracy regions
#        - optional: accuracy window (e.g., 0.83~0.84)
#     """
#     master_path = outdir_root / "all_models_summary.tsv"
#     ensure_master_header(master_path)
#     tested = load_tested(tested_log)

#     all_ckpts = discover_ckpts(models_dir, step_mod=1)
#     coarse_ckpts = discover_ckpts(models_dir, step_mod=max(1, int(coarse_step_mod)))
#     if fine_acc_min is not None and fine_acc_max is not None:
#         logging.info(
#             "两阶段评测开始: coarse_step_mod=%d, fine_step_mod=%d, fine_acc_range=[%.6f, %.6f], 总ckpt=%d, coarse_ckpt=%d",
#             int(coarse_step_mod),
#             int(fine_step_mod),
#             float(min(fine_acc_min, fine_acc_max)),
#             float(max(fine_acc_min, fine_acc_max)),
#             len(all_ckpts),
#             len(coarse_ckpts),
#         )
#     else:
#         logging.info(
#             "两阶段评测开始: coarse_step_mod=%d, fine_step_mod=%d, stable_acc_delta=%.6f, 总ckpt=%d, coarse_ckpt=%d",
#             int(coarse_step_mod),
#             int(fine_step_mod),
#             float(stable_acc_delta),
#             len(all_ckpts),
#             len(coarse_ckpts),
#         )

#     coarse_results: List[dict] = []

#     # Stage 1: coarse checkpoints
#     for p in coarse_ckpts:
#         sp = str(p.resolve())
#         model_tag = p.stem
#         step = _extract_last_int(model_tag)
#         if sp in tested:
#             cached_acc = get_model_avg_acc_from_master(master_path, model_tag)
#             if cached_acc is not None:
#                 coarse_results.append({"path": p, "model_tag": model_tag, "step": step, "avg_acc": float(cached_acc)})
#             continue

#         t0 = time.perf_counter()
#         model_tag, total, avg_acc, total_t, avg_t, avg_tr = evaluate_model(
#             model_path=sp,
#             outdir_root=outdir_root,
#             evaluator_pool=evaluator_pool,
#             dirs=dirs,
#         )
#         append_master(master_path, model_tag, total, avg_acc, total_t, avg_t, avg_tr)
#         tested.add(sp)
#         append_tested(tested_log, sp)
#         coarse_results.append({"path": p, "model_tag": model_tag, "step": step, "avg_acc": float(avg_acc)})
#         logging.info("[Coarse %s] Done in %.2fs", model_tag, time.perf_counter() - t0)

#     # Stage 2: fine checkpoints
#     if fine_acc_min is not None and fine_acc_max is not None:
#         fine_candidates = select_fine_candidates_from_acc_window(
#             coarse_results=coarse_results,
#             all_ckpts=all_ckpts,
#             fine_step_mod=max(1, int(fine_step_mod)),
#             fine_acc_min=float(fine_acc_min),
#             fine_acc_max=float(fine_acc_max),
#         )
#         fine_reason = f"精度窗口[{min(fine_acc_min, fine_acc_max):.4f}, {max(fine_acc_min, fine_acc_max):.4f}]"
#     else:
#         fine_candidates = select_fine_candidates_from_stable_regions(
#             coarse_results=coarse_results,
#             all_ckpts=all_ckpts,
#             fine_step_mod=max(1, int(fine_step_mod)),
#             stable_acc_delta=float(stable_acc_delta),
#         )
#         fine_reason = f"稳定区间(delta<={stable_acc_delta:.6f})"
#     fine_candidates = [p for p in fine_candidates if str(p.resolve()) not in tested]

#     logging.info("两阶段评测: %s 细评候选=%d", fine_reason, len(fine_candidates))
#     for p in fine_candidates:
#         sp = str(p.resolve())
#         t0 = time.perf_counter()
#         model_tag, total, avg_acc, total_t, avg_t, avg_tr = evaluate_model(
#             model_path=sp,
#             outdir_root=outdir_root,
#             evaluator_pool=evaluator_pool,
#             dirs=dirs,
#         )
#         append_master(master_path, model_tag, total, avg_acc, total_t, avg_t, avg_tr)
#         tested.add(sp)
#         append_tested(tested_log, sp)
#         logging.info("[Fine %s] Done in %.2fs", model_tag, time.perf_counter() - t0)

#     best = write_best_result(master_path, best_result_path)
#     if best is not None:
#         logging.info(
#             "best_result 已更新: %s (acc=%.6f) -> %s",
#             best["model_name"],
#             best["average_accuracy"],
#             str(best_result_path),
#         )
#     else:
#         logging.info("best_result 未生成: 目前没有有效平均精度。")

#     return master_path


# def loop_eval_new_ckpts(
#     models_dir: Path,
#     outdir_root: Path,
#     poll_sec: float,
#     stable_sec: float,
#     idle_exit_sec: Optional[float],
#     step_mod: int,
#     tested_log: Path,
#     evaluator_pool: PersistentEvaluatorPool,
#     dirs: List[Path],
# ) -> Path:
#     master_path = outdir_root / "all_models_summary.tsv"
#     ensure_master_header(master_path)
#     tested = load_tested(tested_log)
#     last_sizes: dict[str, int] = {}
#     last_new_ts = time.time()

#     logging.info(
#         "进入在线评测模式: models_dir=%s, poll_sec=%.1f, stable_sec=%.1f, step_mod=%d, idle_exit_sec=%s",
#         str(models_dir),
#         poll_sec,
#         stable_sec,
#         step_mod,
#         str(idle_exit_sec),
#     )
#     logging.info("已加载历史已评测ckpt数: %d", len(tested))

#     while True:
#         ckpts = discover_ckpts(models_dir, step_mod=step_mod)
#         candidates = []
#         for p in ckpts:
#             sp = str(p.resolve())
#             if sp in tested:
#                 continue
#             if is_file_stable(p, last_sizes, stable_sec=stable_sec):
#                 candidates.append(p)

#         if candidates:
#             logging.info("发现 %d 个新模型待评测: %s", len(candidates), " -> ".join([c.stem for c in candidates]))
#             for p in candidates:
#                 sp = str(p.resolve())
#                 t0 = time.perf_counter()
#                 model_tag, total, avg_acc, total_t, avg_t, avg_tr = evaluate_model(
#                     model_path=sp,
#                     outdir_root=outdir_root,
#                     evaluator_pool=evaluator_pool,
#                     dirs=dirs,
#                 )
#                 append_master(master_path, model_tag, total, avg_acc, total_t, avg_t, avg_tr)
#                 tested.add(sp)
#                 append_tested(tested_log, sp)
#                 logging.info("[%s] Done in %.2fs", model_tag, time.perf_counter() - t0)
#             last_new_ts = time.time()
#         else:
#             if idle_exit_sec is not None and (time.time() - last_new_ts) > idle_exit_sec:
#                 logging.info("超过 %.0fs 没有新checkpoint，退出。", idle_exit_sec)
#                 break
#             time.sleep(poll_sec)

#     return master_path


# def evaluate_once(
#     model_path: str,
#     outdir_root: Path,
#     evaluator_pool: PersistentEvaluatorPool,
#     dirs: List[Path],
# ) -> Path:
#     master_path = outdir_root / "all_models_summary.tsv"
#     ensure_master_header(master_path)
#     t0 = time.perf_counter()
#     model_tag, total, avg_acc, total_t, avg_t, avg_tr = evaluate_model(
#         model_path=model_path,
#         outdir_root=outdir_root,
#         evaluator_pool=evaluator_pool,
#         dirs=dirs,
#     )
#     append_master(master_path, model_tag, total, avg_acc, total_t, avg_t, avg_tr)
#     logging.info("[%s] Done in %.2fs", model_tag, time.perf_counter() - t0)
#     return master_path


# def parse_args():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model_path", type=str, default=None, help="单个模型路径（与 --models_dir 互斥）")
#     ap.add_argument("--models_dir", type=str, default=None, help="checkpoint目录，开启在线轮询评测")
#     ap.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
#     ap.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
#     ap.add_argument("--poll_sec", type=float, default=30.0, help="轮询间隔秒")
#     ap.add_argument("--stable_sec", type=float, default=10.0, help="checkpoint最小稳定时长秒")
#     ap.add_argument("--idle_exit_sec", type=float, default=None, help="超过该秒数无新ckpt则退出；默认一直运行")
#     ap.add_argument("--step_mod", type=int, default=1, help="仅评测step%%step_mod==0的checkpoint；1表示不过滤")
#     ap.add_argument("--clf_n_estimators", type=int, default=32, help="TabICLClassifier n_estimators")
#     ap.add_argument(
#         "--clf_batch_size",
#         type=int,
#         default=8,
#         help="TabICLClassifier batch_size；设为-1表示None（单次处理所有ensemble）",
#     )
#     ap.add_argument("--clf_n_jobs", type=int, default=1, help="TabICLClassifier n_jobs（推荐1避免8进程CPU争抢）")
#     ap.add_argument("--cpu_threads", type=int, default=1, help="每个GPU进程的CPU线程上限")
#     ap.add_argument("--use_torch_compile", action="store_true", help="启用 torch.compile（默认关闭）")
#     ap.add_argument(
#         "--torch_compile_mode",
#         type=str,
#         default="reduce-overhead",
#         help="torch.compile mode，设为none表示使用PyTorch默认",
#     )
#     ap.add_argument("--torch_compile_backend", type=str, default=None, help="torch.compile backend，默认PyTorch自动")
#     ap.add_argument("--torch_compile_fullgraph", action="store_true", help="torch.compile fullgraph=True")
#     ap.add_argument("--torch_compile_dynamic", action="store_true", help="torch.compile dynamic=True")
#     ap.add_argument(
#         "--torchinductor_cache_dir",
#         type=str,
#         default=None,
#         help="TORCHINDUCTOR_CACHE_DIR，默认 <outdir>/_torchinductor_cache",
#     )
#     ap.add_argument(
#         "--cache_root",
#         type=str,
#         default=None,
#         help="预处理数据缓存目录，默认 <outdir>/_dataset_cache",
#     )
#     ap.add_argument(
#         "--tested_log",
#         type=str,
#         default=None,
#         help="已评测checkpoint记录文件，默认 <outdir>/tested_ckpts.txt",
#     )
#     ap.add_argument("--two_stage", action="store_true", help="两阶段模式：先粗评再在稳定区间细评")
#     ap.add_argument("--coarse_step_mod", type=int, default=100, help="两阶段粗评步长筛选（默认100）")
#     ap.add_argument("--fine_step_mod", type=int, default=50, help="两阶段细评步长筛选（默认50）")
#     ap.add_argument("--fine_acc_min", type=float, default=0.825, help="两阶段细评精度下界（与 --fine_acc_max 配合）")
#     ap.add_argument("--fine_acc_max", type=float, default=0.84, help="两阶段细评精度上界（与 --fine_acc_min 配合）")
#     ap.add_argument(
#         "--stable_acc_delta",
#         type=float,
#         default=0.001,
#         help="判定“稳定区间”的相邻粗评精度差阈值（默认0.001；若设置fine_acc窗口则该参数不使用）",
#     )
#     ap.add_argument(
#         "--best_result_path",
#         type=str,
#         default=None,
#         help="best_result输出文件，默认 <outdir>/best_result.tsv",
#     )
#     return ap.parse_args()


# def main():
#     logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
#     args = parse_args()

#     data_root = Path(args.data_root)
#     outdir_root = Path(args.outdir)
#     outdir_root.mkdir(parents=True, exist_ok=True)

#     tested_log = Path(args.tested_log) if args.tested_log else (outdir_root / "tested_ckpts.txt")
#     best_result_path = Path(args.best_result_path) if args.best_result_path else (outdir_root / "best_result.tsv")
#     clf_batch_size = None if int(args.clf_batch_size) <= 0 else int(args.clf_batch_size)
#     clf_n_jobs = None if int(args.clf_n_jobs) <= 0 else int(args.clf_n_jobs)
#     cpu_threads = max(1, int(args.cpu_threads))
#     cache_root = Path(args.cache_root) if args.cache_root else (outdir_root / "_dataset_cache")
#     use_torch_compile = bool(args.use_torch_compile)
#     torch_compile_mode = args.torch_compile_mode
#     if torch_compile_mode is not None and torch_compile_mode.strip().lower() in {"", "none"}:
#         torch_compile_mode = None
#     torch_compile_backend = args.torch_compile_backend
#     if torch_compile_backend is not None and torch_compile_backend.strip().lower() in {"", "none"}:
#         torch_compile_backend = None
#     torch_compile_fullgraph = bool(args.torch_compile_fullgraph)
#     torch_compile_dynamic = True if args.torch_compile_dynamic else None
#     torchinductor_cache_dir = (
#         str(Path(args.torchinductor_cache_dir))
#         if args.torchinductor_cache_dir
#         else str(outdir_root / "_torchinductor_cache")
#     )

#     dirs = [d for d in sorted(data_root.iterdir()) if d.is_dir()]
#     summarize_task_types(dirs)

#     gpu_devices = resolve_gpu_devices(FIXED_GPUS)
#     if not gpu_devices:
#         raise RuntimeError("未检测到可用GPU设备，无法执行并行评测。")
#     logging.info("常驻评测池使用 %d 张GPU: %s", len(gpu_devices), ",".join(gpu_devices))
#     if use_torch_compile:
#         logging.info(
#             "torch.compile 已开启: mode=%s, backend=%s, fullgraph=%s, dynamic=%s, cache=%s",
#             str(torch_compile_mode),
#             str(torch_compile_backend),
#             str(torch_compile_fullgraph),
#             str(torch_compile_dynamic),
#             torchinductor_cache_dir,
#         )

#     evaluator_pool = PersistentEvaluatorPool(
#         gpu_devices=gpu_devices,
#         cache_root=cache_root,
#         clf_n_estimators=int(args.clf_n_estimators),
#         clf_batch_size=clf_batch_size,
#         clf_n_jobs=clf_n_jobs,
#         cpu_threads=cpu_threads,
#         use_torch_compile=use_torch_compile,
#         torch_compile_mode=torch_compile_mode,
#         torch_compile_backend=torch_compile_backend,
#         torch_compile_fullgraph=torch_compile_fullgraph,
#         torch_compile_dynamic=torch_compile_dynamic,
#         torchinductor_cache_dir=torchinductor_cache_dir,
#     )

#     try:
#         if args.two_stage:
#             if not args.models_dir:
#                 raise ValueError("--two_stage 需要同时提供 --models_dir")
#             if (args.fine_acc_min is None) ^ (args.fine_acc_max is None):
#                 raise ValueError("--fine_acc_min 和 --fine_acc_max 需要同时设置，或都不设置")
#             models_dir = Path(args.models_dir)
#             if not models_dir.exists():
#                 raise FileNotFoundError(f"--models_dir 不存在: {models_dir}")
#             master_path = evaluate_two_stage_ckpts(
#                 models_dir=models_dir,
#                 outdir_root=outdir_root,
#                 tested_log=tested_log,
#                 evaluator_pool=evaluator_pool,
#                 dirs=dirs,
#                 coarse_step_mod=max(1, int(args.coarse_step_mod)),
#                 fine_step_mod=max(1, int(args.fine_step_mod)),
#                 stable_acc_delta=float(args.stable_acc_delta),
#                 fine_acc_min=None if args.fine_acc_min is None else float(args.fine_acc_min),
#                 fine_acc_max=None if args.fine_acc_max is None else float(args.fine_acc_max),
#                 best_result_path=best_result_path,
#             )
#             print("\n汇总总表：", master_path)
#             print("best_result：", best_result_path)
#             return

#         if args.models_dir:
#             models_dir = Path(args.models_dir)
#             if not models_dir.exists():
#                 raise FileNotFoundError(f"--models_dir 不存在: {models_dir}")
#             master_path = loop_eval_new_ckpts(
#                 models_dir=models_dir,
#                 outdir_root=outdir_root,
#                 poll_sec=float(args.poll_sec),
#                 stable_sec=float(args.stable_sec),
#                 idle_exit_sec=None if args.idle_exit_sec is None else float(args.idle_exit_sec),
#                 step_mod=int(args.step_mod),
#                 tested_log=tested_log,
#                 evaluator_pool=evaluator_pool,
#                 dirs=dirs,
#             )
#             best = write_best_result(outdir_root / "all_models_summary.tsv", best_result_path)
#             if best is not None:
#                 print("best_result：", best_result_path)
#             print("\n汇总总表：", master_path)
#             return

#         model_path = args.model_path or DEFAULT_MODEL_PATH
#         master_path = evaluate_once(
#             model_path=model_path,
#             outdir_root=outdir_root,
#             evaluator_pool=evaluator_pool,
#             dirs=dirs,
#         )
#         best = write_best_result(outdir_root / "all_models_summary.tsv", best_result_path)
#         if best is not None:
#             print("best_result：", best_result_path)
#         print("\n汇总总表：", master_path)
#     finally:
#         evaluator_pool.close()


# if __name__ == "__main__":
#     main()
