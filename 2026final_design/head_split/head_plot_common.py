"""
封头论文图绘制公共模块。

功能：
1. 统一导入包、路径、颜色、字体、参数名等全局配置；
2. 读取 head.csv，并生成 feasibility/stage/mass_reduction 等派生字段；
3. 提供所有绘图脚本共享的坐标轴美化、标注、保存、GPR 评估和敏感性分析函数。

使用方式：
- 不建议单独运行本文件；
- 每张 PNG 对应一个 plot_*.py 脚本；
- 批量出图运行 head_figures_split_by_png.py。
"""

from __future__ import annotations

import importlib
import os
import re
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parent
MPL_CONFIG_DIR = ROOT / ".matplotlib_cache"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.colors import Normalize
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


# -----------------------------------------------------------------------------
# 自包含 GPR 设置
# -----------------------------------------------------------------------------
LENGTH_SCALE_GRID = np.array([0.35, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 5.00], dtype=float)
NOISE = 1.0e-6


class SimpleGaussianProcess:
    """轻量级 RBF 高斯过程回归模型，用于论文图中的代理模型展示。"""

    def __init__(self, length_scale_grid: np.ndarray | list[float] = LENGTH_SCALE_GRID, noise: float = NOISE) -> None:
        self.length_scale_grid = np.asarray(length_scale_grid, dtype=float)
        self.noise = float(noise)
        self.length_scale_: float | None = None
        self.x_mean_: np.ndarray | None = None
        self.x_std_: np.ndarray | None = None
        self.y_mean_: float | None = None
        self.y_std_: float | None = None
        self.x_train_: np.ndarray | None = None
        self.lower_cholesky_: np.ndarray | None = None
        self.alpha_: np.ndarray | None = None

    @staticmethod
    def _safe_std(values: np.ndarray) -> np.ndarray:
        std = values.std(axis=0, ddof=0)
        return np.where(std < 1.0e-12, 1.0, std)

    @staticmethod
    def _rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float) -> np.ndarray:
        diff = x1[:, None, :] - x2[None, :, :]
        sq_dist = np.sum(diff * diff, axis=2)
        return np.exp(-0.5 * sq_dist / (length_scale**2))

    def _standardize_x(self, x: np.ndarray) -> np.ndarray:
        if self.x_mean_ is None or self.x_std_ is None:
            raise RuntimeError("SimpleGaussianProcess has not been fitted yet.")
        return (np.asarray(x, dtype=float) - self.x_mean_) / self.x_std_

    def _fit_with_length_scale(self, x_scaled: np.ndarray, y_scaled: np.ndarray, length_scale: float) -> tuple[float, np.ndarray, np.ndarray]:
        n = len(x_scaled)
        kernel = self._rbf_kernel(x_scaled, x_scaled, length_scale)
        kernel = kernel + (self.noise + 1.0e-10) * np.eye(n)

        last_error: Exception | None = None
        for jitter in (0.0, 1.0e-9, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5):
            try:
                lower = np.linalg.cholesky(kernel + jitter * np.eye(n))
                alpha = np.linalg.solve(lower.T, np.linalg.solve(lower, y_scaled))
                nll = 0.5 * float(y_scaled @ alpha)
                nll += float(np.sum(np.log(np.diag(lower))))
                nll += 0.5 * n * np.log(2.0 * np.pi)
                return nll, lower, alpha
            except np.linalg.LinAlgError as exc:
                last_error = exc
        raise RuntimeError("GPR kernel matrix is not positive definite.") from last_error

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SimpleGaussianProcess":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if x.ndim != 2:
            raise ValueError("x must be a 2D array.")
        if len(x) != len(y):
            raise ValueError("x and y must have the same number of rows.")

        self.x_mean_ = x.mean(axis=0)
        self.x_std_ = self._safe_std(x)
        x_scaled = (x - self.x_mean_) / self.x_std_

        self.y_mean_ = float(y.mean())
        y_std = float(y.std(ddof=0))
        self.y_std_ = y_std if y_std > 1.0e-12 else 1.0
        y_scaled = (y - self.y_mean_) / self.y_std_

        best = None
        for length_scale in self.length_scale_grid:
            try:
                candidate = self._fit_with_length_scale(x_scaled, y_scaled, float(length_scale))
            except RuntimeError:
                continue
            if best is None or candidate[0] < best[0]:
                best = (candidate[0], float(length_scale), candidate[1], candidate[2])

        if best is None:
            raise RuntimeError("Could not fit SimpleGaussianProcess with the provided length_scale_grid.")

        _, self.length_scale_, self.lower_cholesky_, self.alpha_ = best
        self.x_train_ = x_scaled
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.x_train_ is None or self.lower_cholesky_ is None or self.alpha_ is None:
            raise RuntimeError("SimpleGaussianProcess has not been fitted yet.")
        if self.length_scale_ is None or self.y_mean_ is None or self.y_std_ is None:
            raise RuntimeError("SimpleGaussianProcess internal parameters are incomplete.")

        x_scaled = self._standardize_x(np.asarray(x, dtype=float))
        k_trans = self._rbf_kernel(x_scaled, self.x_train_, self.length_scale_)
        mean_scaled = k_trans @ self.alpha_

        v = np.linalg.solve(self.lower_cholesky_, k_trans.T)
        var_scaled = np.maximum(1.0 - np.sum(v * v, axis=0), 0.0)

        mean = mean_scaled * self.y_std_ + self.y_mean_
        std = np.sqrt(var_scaled) * self.y_std_
        return mean, std


CSV_PATH = ROOT / "head.csv"
OUT_DIR = ROOT / "figures" / "head"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 600

COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#222222",
    "gray": "#8A8F98",
    "light_gray": "#D8DDE3",
    "pale_green": "#DDEFE7",
    "pale_red": "#F6E1DA",
}

PARAMETER_COLS = [
    "thickness_shell",
    "longitude_num",
    "latitude_num",
    "longitude_height",
    "latitude_height",
    "longitude_thickness",
    "latitude_thickness",
    "longitude_rib_boundary_angle",
    "top_unribbed_cap_angle",
]

PARAMETER_LABELS = {
    "thickness_shell": "壳厚 t",
    "longitude_num": "经向数",
    "latitude_num": "纬向数",
    "longitude_height": "经向高度",
    "latitude_height": "纬向高度",
    "longitude_thickness": "经向厚度",
    "latitude_thickness": "纬向厚度",
    "longitude_rib_boundary_angle": "经向边界角",
    "top_unribbed_cap_angle": "顶部无筋角",
}

RESPONSE_COLS = ["total_mass", "buckling_pressure"]
RESPONSE_LABELS = {
    "total_mass": "质量（kg）",
    "buckling_pressure": "屈曲特征值",
}

INTEGER_PARAMETER_COLS = {
    "longitude_num",
    "latitude_num",
    "longitude_rib_boundary_angle",
    "top_unribbed_cap_angle",
}

GPR_RANDOM_SEED = 20260510
INITIAL_SCREENING_LAST_CASE = 45

CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
    "DejaVu Sans",
]


def available_chinese_fonts() -> list[str]:
    available = {font.name for font in font_manager.fontManager.ttflist}
    matched = [font for font in CHINESE_FONT_CANDIDATES if font in available]
    return matched or ["DejaVu Sans"]


def try_enable_nature_style() -> str:
    try:
        importlib.import_module("scienceplots")
        plt.style.use(["science", "nature", "no-latex"])
        style_name = "SciencePlots 中文论文风格"
    except Exception:
        style_name = "自定义中文论文风格"

    chinese_fonts = available_chinese_fonts()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": chinese_fonts,
            "axes.unicode_minus": False,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "axes.linewidth": 0.8,
            "axes.edgecolor": COLORS["black"],
            "axes.labelcolor": COLORS["black"],
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 6.2,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": DPI,
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "lines.linewidth": 1.15,
            "patch.linewidth": 0.8,
        }
    )
    return style_name


def case_index(case_name: object) -> int:
    text = str(case_name)
    if text == "0":
        return 0
    match = re.match(r"G(\d+)", text)
    if match:
        return int(match.group(1))
    return -1


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, float]:
    df = pd.read_csv(CSV_PATH)
    numeric_cols = [
        "thickness_shell",
        "longitude_num",
        "latitude_num",
        "rib_levels",
        "longitude_height",
        "latitude_height",
        "longitude_thickness",
        "latitude_thickness",
        "longitude_rib_boundary_angle",
        "top_unribbed_cap_angle",
        "total_mass",
        "eigenvalue",
        "buckling_pressure",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["case_index"] = df["case_name"].map(case_index)
    df = df.sort_values("case_index").reset_index(drop=True)
    baseline = df.iloc[0].copy()
    baseline["total_mass"] = float(baseline["total_mass"]) * 1000.0
    limit = float(baseline["buckling_pressure"])

    design = df.iloc[1:].copy()
    design["feasible"] = (design["buckling_pressure"] > limit) & (design["eigenvalue"] > limit)
    design["total_mass"] = design["total_mass"] * 1000.0
    design["mass_reduction"] = 1.0 - design["total_mass"] / (float(baseline["total_mass"]) * 1000.0)
    design["buckling_margin"] = design["buckling_pressure"] - limit
    design["stage"] = np.where(
        design["case_index"] <= INITIAL_SCREENING_LAST_CASE,
        "前期筛选",
        "后期验证",
    )

    feasible = design[design["feasible"]].copy()
    if feasible.empty:
        raise RuntimeError("No feasible head design found.")

    best = feasible.sort_values("total_mass").iloc[0].copy()
    return df, baseline, design, best, limit


def running_best(design: pd.DataFrame) -> pd.DataFrame:
    ordered = design.sort_values("case_index").copy()
    ordered["running_best_mass"] = np.nan
    best_value = np.inf
    for idx, row in ordered.iterrows():
        if bool(row["feasible"]) and row["total_mass"] < best_value:
            best_value = float(row["total_mass"])
        if np.isfinite(best_value):
            ordered.loc[idx, "running_best_mass"] = best_value
    return ordered


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )


def beautify_axis(ax: plt.Axes, minor: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", top=False, right=False)
    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(False)


def annotate_best(ax: plt.Axes, x: float, y: float, text: str, xytext: tuple[float, float]) -> None:
    annotation = ax.annotate(
        text,
        xy=(x, y),
        xytext=xytext,
        textcoords="offset points",
        ha="left",
        va="bottom",
        arrowprops={
            "arrowstyle": "-",
            "color": COLORS["black"],
            "lw": 0.7,
            "shrinkA": 0,
            "shrinkB": 3,
        },
        fontsize=6.4,
        color=COLORS["black"],
    )
    annotation.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight", pad_inches=0.03)
    for temp_file in OUT_DIR.glob("*.ldtmp"):
        temp_file.unlink(missing_ok=True)


def cleanup_temp_files() -> None:
    for temp_file in OUT_DIR.glob("*.ldtmp"):
        temp_file.unlink(missing_ok=True)


def cleanup_non_png_outputs() -> None:
    for suffix in ("*.pdf", "*.svg"):
        for artifact in OUT_DIR.glob(suffix):
            artifact.unlink(missing_ok=True)


def binned_median_line(x: pd.Series, y: pd.Series, bins: int = 6) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 4:
        return np.array([]), np.array([])

    unique_x = np.sort(frame["x"].unique())
    if len(unique_x) <= bins:
        grouped = frame.groupby("x", as_index=False)["y"].median().sort_values("x")
        return grouped["x"].to_numpy(), grouped["y"].to_numpy()

    frame["bin"] = pd.qcut(frame["x"], q=min(bins, len(unique_x)), duplicates="drop")
    grouped = frame.groupby("bin", observed=False).agg(x=("x", "median"), y=("y", "median")).dropna()
    return grouped["x"].to_numpy(), grouped["y"].to_numpy()


def normalize_to_unit(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    low = values.min()
    high = values.max()
    if not np.isfinite(low) or not np.isfinite(high) or abs(high - low) < 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - low) / (high - low)


def enforce_integer_parameters(frame: pd.DataFrame) -> pd.DataFrame:
    adjusted = frame.copy()
    for col in INTEGER_PARAMETER_COLS:
        if col in adjusted.columns:
            adjusted[col] = adjusted[col].round().astype(int)
    return adjusted


def physical_feasible(frame: pd.DataFrame) -> np.ndarray:
    return np.ones(len(frame), dtype=bool)


def metric_summary(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = predicted - actual
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
    }


def safe_abs_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """计算绝对皮尔逊相关系数；若任一序列方差为 0，则返回 NaN。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x, ddof=0) < 1.0e-12 or np.std(y, ddof=0) < 1.0e-12:
        return np.nan
    return float(abs(np.corrcoef(x, y)[0, 1]))


def fit_gpr_models(design: pd.DataFrame, targets: list[str]) -> dict[str, SimpleGaussianProcess]:
    x_train = design[PARAMETER_COLS].to_numpy(dtype=float)
    models: dict[str, SimpleGaussianProcess] = {}
    for target in targets:
        model = SimpleGaussianProcess(LENGTH_SCALE_GRID, NOISE)
        model.fit(x_train, design[target].to_numpy(dtype=float))
        models[target] = model
    return models


def cross_validated_gpr_predictions(design: pd.DataFrame, targets: list[str], n_splits: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(GPR_RANDOM_SEED)
    shuffled = rng.permutation(len(design))
    folds = np.array_split(shuffled, n_splits)
    x_all = design[PARAMETER_COLS].to_numpy(dtype=float)

    rows = []
    metrics = []
    for target in targets:
        actual_all = design[target].to_numpy(dtype=float)
        pred_all = np.full(len(design), np.nan)
        std_all = np.full(len(design), np.nan)

        for fold in folds:
            train_idx = np.setdiff1d(np.arange(len(design)), fold)
            model = SimpleGaussianProcess(LENGTH_SCALE_GRID, NOISE)
            model.fit(x_all[train_idx], actual_all[train_idx])
            mean, std = model.predict(x_all[fold])
            pred_all[fold] = mean
            std_all[fold] = std

        summary = metric_summary(actual_all, pred_all)
        metrics.append({"target": target, **summary})
        for row_idx, actual, pred, std in zip(design.index, actual_all, pred_all, std_all):
            rows.append(
                {
                    "case_name": design.loc[row_idx, "case_name"],
                    "case_index": design.loc[row_idx, "case_index"],
                    "target": target,
                    "actual": actual,
                    "predicted": pred,
                    "std": std,
                    "feasible": bool(design.loc[row_idx, "feasible"]),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(metrics)


def random_design_samples(design: pd.DataFrame, n_samples: int, rng: np.random.Generator) -> pd.DataFrame:
    low = design[PARAMETER_COLS].min().to_numpy(dtype=float)
    high = design[PARAMETER_COLS].max().to_numpy(dtype=float)
    sampled = low + rng.random((n_samples, len(PARAMETER_COLS))) * (high - low)
    frame = pd.DataFrame(sampled, columns=PARAMETER_COLS)
    return enforce_integer_parameters(frame)


def gp_permutation_sensitivity(
    design: pd.DataFrame,
    models: dict[str, SimpleGaussianProcess],
    n_samples: int = 5000,
) -> pd.DataFrame:
    rng = np.random.default_rng(GPR_RANDOM_SEED + 17)
    samples = random_design_samples(design, n_samples, rng)
    samples = samples.loc[physical_feasible(samples)].reset_index(drop=True)
    x_base = samples[PARAMETER_COLS].to_numpy(dtype=float)

    rows = []
    for target, model in models.items():
        base_pred, _ = model.predict(x_base)
        effects = []

        for param in PARAMETER_COLS:
            perturbed = samples.copy()
            perturbed[param] = rng.permutation(perturbed[param].to_numpy())
            perturbed = enforce_integer_parameters(perturbed)
            perturbed_pred, _ = model.predict(perturbed[PARAMETER_COLS].to_numpy(dtype=float))
            effect = float(np.sqrt(np.mean((perturbed_pred - base_pred) ** 2)))
            corr = safe_abs_pearson(design[param].to_numpy(dtype=float), design[target].to_numpy(dtype=float))
            effects.append(effect)
            rows.append(
                {
                    "target": target,
                    "parameter": param,
                    "gp_permutation_effect": effect,
                    "abs_pearson_correlation": corr,
                }
            )

        total_effect = sum(effects)
        for row in rows:
            if row["target"] == target:
                row["normalized_gp_effect"] = row["gp_permutation_effect"] / total_effect if total_effect > 1e-12 else np.nan

    return pd.DataFrame(rows)


def _feasible_random_samples(
    design: pd.DataFrame,
    n_samples: int,
    rng: np.random.Generator,
    batch_factor: int = 5,
) -> pd.DataFrame:
    pieces = []
    remaining = int(n_samples)
    while remaining > 0:
        trial_n = max(remaining * batch_factor, 1000)
        trial = random_design_samples(design, trial_n, rng)
        trial = trial.loc[physical_feasible(trial)]
        if not trial.empty:
            pieces.append(trial.iloc[:remaining].copy())
            remaining -= min(len(trial), remaining)
    return pd.concat(pieces, ignore_index=True).iloc[:n_samples].copy()


def sobol_sensitivity_from_gp(
    design: pd.DataFrame,
    models: dict[str, SimpleGaussianProcess],
    n_samples: int = 6000,
) -> pd.DataFrame:
    rng = np.random.default_rng(GPR_RANDOM_SEED + 29)
    a_df = _feasible_random_samples(design, n_samples, rng)
    b_df = _feasible_random_samples(design, n_samples, rng)

    x_a = a_df[PARAMETER_COLS].to_numpy(dtype=float)
    x_b = b_df[PARAMETER_COLS].to_numpy(dtype=float)

    rows = []
    for target, model in models.items():
        y_a, _ = model.predict(x_a)
        y_b, _ = model.predict(x_b)

        var_y = float(np.var(np.concatenate([y_a, y_b]), ddof=1))
        if var_y < 1e-12:
            var_y = np.nan

        for param in PARAMETER_COLS:
            ab_df = a_df.copy()
            ab_df[param] = b_df[param].to_numpy()
            ab_df = enforce_integer_parameters(ab_df)
            valid_mask = physical_feasible(ab_df)
            if valid_mask.sum() < 50:
                s1 = np.nan
                st = np.nan
            else:
                y_a_v = y_a[valid_mask]
                y_b_v = y_b[valid_mask]
                y_ab, _ = model.predict(ab_df.loc[valid_mask, PARAMETER_COLS].to_numpy(dtype=float))
                if not np.isfinite(var_y):
                    s1 = np.nan
                    st = np.nan
                else:
                    s1 = float(np.mean(y_b_v * (y_ab - y_a_v)) / var_y)
                    st = float(0.5 * np.mean((y_a_v - y_ab) ** 2) / var_y)

            rows.append(
                {
                    "target": target,
                    "parameter": param,
                    "sobol_s1": s1,
                    "sobol_st": st,
                }
            )

    sobol_df = pd.DataFrame(rows)
    sobol_df["sobol_s1"] = sobol_df["sobol_s1"].clip(lower=0.0, upper=1.0)
    sobol_df["sobol_st"] = sobol_df["sobol_st"].clip(lower=0.0, upper=1.0)
    return sobol_df


def response_surface_parameters(sensitivity: pd.DataFrame) -> list[str]:
    combined = (
        sensitivity.groupby("parameter", as_index=False)["normalized_gp_effect"]
        .mean()
        .sort_values("normalized_gp_effect", ascending=False)
    )
    selected = combined["parameter"].head(2).tolist()
    if len(selected) < 2:
        selected = [PARAMETER_COLS[0], PARAMETER_COLS[1]]
    return selected
