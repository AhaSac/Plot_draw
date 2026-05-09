"""
筒体优化过程论文图绘制脚本。

重复运行本脚本会覆盖 figures/cylinder 下同名 PNG 图片。CSV 数据表会保留，
用于论文中复核最优解、交叉验证指标和敏感性分析数据来源。

输出图片：
  figures/cylinder/cylinder_optimization_summary.png
    figures/cylinder/cylinder_before_after_comparison.png
  figures/cylinder/cylinder_mass_constraint_map.png
  figures/cylinder/cylinder_parameter_response_grid.png
  figures/cylinder/cylinder_parallel_coordinates.png
  figures/cylinder/cylinder_parameter_evolution_heatmap.png
  figures/cylinder/cylinder_sampling_scatter_matrix.png
  figures/cylinder/cylinder_gpr_predicted_vs_actual.png
  figures/cylinder/cylinder_gpr_response_surface.png
  figures/cylinder/cylinder_gpr_sensitivity_bar.png
    figures/cylinder/cylinder_gpr_sobol_bar.png
    figures/cylinder/cylinder_viz_tradeoff.png
"""

from __future__ import annotations

import os
import re
import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MPL_CONFIG_DIR = ROOT / ".matplotlib_cache"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.colors import Normalize
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, PercentFormatter

from cylinder_gpr_optimization import LENGTH_SCALE_GRID, NOISE, SimpleGaussianProcess


CSV_PATH = ROOT / "cylinder.csv"
OUT_DIR = ROOT / "figures" / "cylinder"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CYLINDER_LENGTH = 2650.0
INITIAL_LHS_LAST_CASE = 27
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

PARAMETER_COLS = ["t", "n", "p", "a"]
PARAMETER_LABELS = {
    "t": "厚度 t",
    "n": "波纹数 n",
    "p": "波距 p",
    "a": "波幅 a",
}

RESPONSE_COLS = ["total_mass", "buckling_pressure"]
RESPONSE_LABELS = {
    "total_mass": "总质量",
    "buckling_pressure": "屈曲压力",
}

INTEGER_PARAMETER_COLS = {"n", "p", "a"}
GPR_RANDOM_SEED = 20260508

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

VIZ_CASE_DATA = [
    ("G51", 2.801, 11, 214, 51, 0.1190, 0.0160),
    ("G13", 2.8, 11, 225, 70, 0.1276, 0.0603),
    ("G17", 2.8, 11, 220, 80, 0.1333, 0.0445),
    ("G12", 3.0, 11, 225, 70, 0.1367, 0.0885),
    ("G16", 3.0, 11, 220, 80, 0.1429, 0.0776),
    ("G27", 3.5, 9, 270, 60, 0.1474, 0.0518),
    ("G25", 3.0, 12, 200, 80, 0.1475, 0.0543),
    ("G19", 3.5, 11, 225, 50, 0.1478, 0.0406),
    ("G21", 3.5, 10, 240, 60, 0.1505, 0.0792),
    ("G24", 3.5, 12, 200, 60, 0.1571, 0.1499),
    ("G11", 3.5, 11, 225, 70, 0.1596, 0.1683),
    ("G22", 3.5, 10, 240, 80, 0.1618, 0.1468),
    ("G15", 3.5, 11, 220, 80, 0.1667, 0.1806),
    ("G26", 4.0, 9, 270, 60, 0.1686, 0.1072),
    ("G18", 4.0, 11, 225, 50, 0.1690, 0.0714),
    ("G20", 4.0, 10, 240, 60, 0.1721, 0.1434),
    ("G23", 4.0, 12, 200, 60, 0.1796, 0.2333),
    ("G10", 4.0, 11, 225, 70, 0.1824, 0.2609),
    ("G14", 4.0, 11, 220, 80, 0.1906, 0.3152),
    ("G03", 6.0, 11, 225, 70, 0.2739, 0.7081),
    ("G08", 6.0, 11, 220, 80, 0.2862, 0.8339),
    ("G05", 6.5, 9, 280, 75, 0.2863, 0.6923),
    ("G04", 7.0, 9, 260, 60, 0.2961, 0.4098),
    ("G09", 6.5, 7, 360, 110, 0.2961, 0.5535),
    ("G02", 7.0, 8, 305, 75, 0.3011, 0.7031),
    ("G06", 7.5, 7, 350, 105, 0.3382, 0.8192),
    ("G07", 8.0, 6, 395, 120, 0.3593, 0.8019),
    ("G01", 8.0, 8, 300, 100, 0.3703, 1.1609),
]


def available_chinese_fonts() -> list[str]:
    """返回当前系统可用的中文字体候选列表。"""
    """从候选字体中选择当前机器可用的中文字体，避免图片中文变成方块。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    matched = [font for font in CHINESE_FONT_CANDIDATES if font in available]
    return matched or ["DejaVu Sans"]


def try_enable_nature_style() -> str:
    """尝试启用 SciencePlots 风格；失败时回退到自定义风格。"""
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


def draw_viz_tradeoff_figure() -> None:
    """导出 viz.py 中的三联图，作为筒体结果的补充展示。"""
    df = pd.DataFrame(VIZ_CASE_DATA, columns=["name", "t", "n", "p", "a", "mass", "margin"])

    fig = plt.figure(figsize=(15, 10))
    grid = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    ax1 = fig.add_subplot(grid[0, :])
    scatter = ax1.scatter(
        df["mass"],
        df["margin"],
        c=df["t"],
        cmap="viridis",
        s=df["n"] * 15,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
    )

    pareto_rows = []
    sorted_df = df.sort_values("mass").reset_index(drop=True)
    max_margin = -np.inf
    for _, row in sorted_df.iterrows():
        if row["margin"] > max_margin:
            pareto_rows.append(row)
            max_margin = row["margin"]
    pareto_df = pd.DataFrame(pareto_rows)
    ax1.plot(
        pareto_df["mass"],
        pareto_df["margin"],
        "r--",
        alpha=0.5,
        linewidth=1.5,
        label="Pareto front",
        zorder=1,
    )

    for _, row in df.iterrows():
        if row["name"] in {"G51", "G13", "G12", "G11", "G24", "G14", "G03", "G01"}:
            ax1.annotate(
                row["name"],
                (row["mass"], row["margin"]),
                fontsize=8,
                xytext=(5, 5),
                textcoords="offset points",
                color="#444",
            )

    ax1.axhline(y=0, color="red", linestyle=":", alpha=0.4, linewidth=1)
    ax1.axhline(y=0.05, color="orange", linestyle=":", alpha=0.4, linewidth=1)
    ax1.text(0.36, 0.06, "safety threshold ~0.05", fontsize=8, color="orange", alpha=0.8)
    fig.colorbar(scatter, ax=ax1, label="thickness t")
    ax1.set_xlabel("Mass (minimize →)", fontsize=11)
    ax1.set_ylabel("Buckling margin (must be ≥ 0)", fontsize=11)
    ax1.set_title("Mass vs Buckling Margin — Pareto front shows best trade-offs", fontsize=12, pad=10)
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(grid[1, 0])
    dims = ["t", "n", "p", "a", "mass"]
    labels = ["t", "n", "p", "a", "mass"]
    norm_df = df[dims].copy()
    for dim in dims:
        norm_df[dim] = (df[dim] - df[dim].min()) / (df[dim].max() - df[dim].min())

    q1 = df["mass"].quantile(0.25)
    q3 = df["mass"].quantile(0.75)
    colors = []
    for mass in df["mass"]:
        if mass <= q1:
            colors.append("#1D9E75")
        elif mass >= q3:
            colors.append("#D85A30")
        else:
            colors.append("#999999")

    x_pos = np.arange(len(dims))
    for idx in range(len(df)):
        ax2.plot(
            x_pos,
            norm_df.iloc[idx].values,
            color=colors[idx],
            alpha=0.6,
            linewidth=1.2 if colors[idx] == "#1D9E75" else 0.7,
        )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_yticks([])
    ax2.set_title("Parallel coordinates — green = low mass (best), red = high mass", fontsize=11, pad=10)
    for xpos in x_pos:
        ax2.axvline(x=xpos, color="gray", alpha=0.25, linewidth=0.5)
    for xpos, dim in enumerate(dims):
        ax2.text(xpos, 1.05, f"{df[dim].max():.2f}", ha="center", fontsize=7, color="#666")
        ax2.text(xpos, -0.05, f"{df[dim].min():.2f}", ha="center", va="top", fontsize=7, color="#666")

    ax3 = fig.add_subplot(grid[1, 1])
    correlations = {var: df[var].corr(df["mass"]) for var in ["t", "n", "p", "a"]}
    vars_list = list(correlations.keys())
    corrs = [correlations[var] for var in vars_list]
    colors_bar = ["#D85A30" if corr > 0 else "#1D9E75" for corr in corrs]

    bars = ax3.barh(vars_list, corrs, color=colors_bar, alpha=0.8, edgecolor="white")
    for bar, corr in zip(bars, corrs):
        ax3.text(
            corr + (0.02 if corr > 0 else -0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{corr:+.2f}",
            va="center",
            ha="left" if corr > 0 else "right",
            fontsize=10,
            fontweight="bold",
        )
    ax3.axvline(x=0, color="black", linewidth=0.5)
    ax3.set_xlabel("Correlation with mass", fontsize=11)
    ax3.set_xlim(-1, 1)
    ax3.set_title("Which variable drives mass the most?", fontsize=11, pad=10)
    ax3.grid(True, alpha=0.3, axis="x")

    fig.suptitle("Buckling Optimization — 28 Design Cases", fontsize=14, fontweight="bold", y=1.00)
    fig.savefig(OUT_DIR / "cylinder_viz_tradeoff.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def case_index(case_name: object) -> int:
    """将 case_name 解析为整数编号，便于按迭代顺序排序。"""
    text = str(case_name)
    if text == "0":
        return 0

    match = re.match(r"G(\d+)", text)
    if match:
        return int(match.group(1))
    return -1


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, float]:
    """读取原始数据并构建可行性、阶段、收益等派生字段。"""
    # 读取原始 CSV，并生成后续绘图所需的可行性、阶段和优化收益字段。
    df = pd.read_csv(CSV_PATH)
    numeric_cols = ["t", "n", "p", "a", "total_mass", "eigenvalue", "buckling_pressure"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["case_index"] = df["case_name"].map(case_index)
    df = df.sort_values("case_index").reset_index(drop=True)
    baseline = df.iloc[0].copy()
    limit = float(baseline["buckling_pressure"])

    design = df.iloc[1:].copy()
    design["n_times_p"] = design["n"] * design["p"]
    design["length_ok"] = design["n_times_p"] <= CYLINDER_LENGTH
    design["pressure_ok"] = design["buckling_pressure"] > limit
    design["eigen_ok"] = design["eigenvalue"] > limit
    design["feasible"] = design["length_ok"] & design["pressure_ok"] & design["eigen_ok"]
    design["mass_reduction"] = 1.0 - design["total_mass"] / float(baseline["total_mass"])
    design["buckling_margin"] = design["buckling_pressure"] - limit
    design["stage"] = np.where(
        design["case_index"] <= INITIAL_LHS_LAST_CASE,
        "LHS screening",
        "GPR-guided verification",
    )

    feasible = design[design["feasible"]].copy()
    if feasible.empty:
        raise RuntimeError("No feasible cylinder design found.")

    best = feasible.sort_values("total_mass").iloc[0].copy()
    return df, baseline, design, best, limit


def running_best(design: pd.DataFrame) -> pd.DataFrame:
    """按算例顺序计算“截至当前的最优可行质量”曲线。"""
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
    """在多子图中添加 a/b/c/d 面板编号。"""
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
    """统一坐标轴外观，保证论文图风格一致。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", top=False, right=False)
    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(False)


def annotate_best(ax: plt.Axes, x: float, y: float, text: str, xytext: tuple[float, float]) -> None:
    """给最优点加带箭头标注。"""
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
    """保存 PNG 并清理临时文件。"""
    # 只导出 PNG；重复运行会覆盖同名图片。
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight", pad_inches=0.03)
    for temp_file in OUT_DIR.glob("*.ldtmp"):
        temp_file.unlink(missing_ok=True)


def cleanup_temp_files() -> None:
    """清理 matplotlib 遗留临时文件。"""
    for temp_file in OUT_DIR.glob("*.ldtmp"):
        temp_file.unlink(missing_ok=True)


def cleanup_non_png_outputs() -> None:
    """删除旧版导出的 PDF/SVG，仅保留 PNG 图片。"""
    for suffix in ("*.pdf", "*.svg"):
        for artifact in OUT_DIR.glob(suffix):
            artifact.unlink(missing_ok=True)


def binned_median_line(x: pd.Series, y: pd.Series, bins: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """基于分箱中位数构造稳健趋势线，减少离群点影响。"""
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
    """将序列缩放到 [0, 1]，用于平行坐标与热图展示。"""
    values = pd.to_numeric(series, errors="coerce")
    low = values.min()
    high = values.max()
    if not np.isfinite(low) or not np.isfinite(high) or abs(high - low) < 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - low) / (high - low)


def enforce_integer_parameters(frame: pd.DataFrame) -> pd.DataFrame:
    """强制离散设计变量 n/p/a 为整数。"""
    adjusted = frame.copy()
    for col in INTEGER_PARAMETER_COLS:
        if col in adjusted.columns:
            adjusted[col] = adjusted[col].round().astype(int)
    return adjusted


def physical_feasible(frame: pd.DataFrame) -> np.ndarray:
    """物理约束：n * p <= CYLINDER_LENGTH。"""
    if {"n", "p"}.issubset(frame.columns):
        return (frame["n"].to_numpy(dtype=float) * frame["p"].to_numpy(dtype=float)) <= CYLINDER_LENGTH
    return np.ones(len(frame), dtype=bool)


def metric_summary(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """计算常用回归误差指标：R2、RMSE、MAE。"""
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


def fit_gpr_models(design: pd.DataFrame, targets: list[str]) -> dict[str, SimpleGaussianProcess]:
    """分别拟合质量和屈曲压力的高斯过程代理模型。"""
    # 使用已有的轻量高斯过程模型拟合目标函数和约束响应。
    x_train = design[PARAMETER_COLS].to_numpy(dtype=float)
    models = {}
    for target in targets:
        model = SimpleGaussianProcess(LENGTH_SCALE_GRID, NOISE)
        model.fit(x_train, design[target].to_numpy(dtype=float))
        models[target] = model
    return models


def cross_validated_gpr_predictions(design: pd.DataFrame, targets: list[str], n_splits: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """五折交叉验证，输出逐样本预测结果和整体误差指标。"""
    # 五折交叉验证：用于绘制“预测值-真实值”图并计算 R2、RMSE、MAE。
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
    """在参数边界内生成随机参考样本，用于采样分布对比。"""
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
    """基于置换效应评估参数敏感性，并给出相关性参考。"""
    # 置换敏感性：打乱单个参数后观察高斯过程预测变化，变化越大说明越敏感。
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
            corr = float(abs(np.corrcoef(design[param].to_numpy(dtype=float), design[target].to_numpy(dtype=float))[0, 1]))
            if not np.isfinite(corr):
                corr = np.nan
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
    """通过拒绝采样生成满足 n*p 物理约束的随机样本。"""
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
    """基于高斯过程代理模型估计 Sobol 一阶指数和总阶指数。"""
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

        for idx, param in enumerate(PARAMETER_COLS):
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
                    # Saltelli 常用估计器：S1 为一阶指数，ST 为总阶指数。
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


def draw_summary_figure(
    baseline: pd.Series,
    design: pd.DataFrame,
    best: pd.Series,
    pressure_limit: float,
) -> None:
    """综合总览图：迭代过程、质量-约束权衡、可行域与阶段最优。"""
    fig = plt.figure(figsize=(7.2, 5.35), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 0.95])

    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[1, 0])
    ax3 = fig.add_subplot(grid[1, 1])

    ordered = running_best(design)
    infeasible = ordered[~ordered["feasible"]]
    feasible = ordered[ordered["feasible"]]

    ax0.scatter(
        infeasible["case_index"],
        infeasible["total_mass"],
        s=17,
        facecolors="white",
        edgecolors=COLORS["gray"],
        linewidths=0.65,
        alpha=0.78,
        label="不可行",
    )
    ax0.scatter(
        feasible["case_index"],
        feasible["total_mass"],
        s=22,
        color=COLORS["green"],
        edgecolors="white",
        linewidths=0.35,
        alpha=0.95,
        label="可行",
        zorder=3,
    )
    ax0.plot(
        ordered["case_index"],
        ordered["running_best_mass"],
        color=COLORS["orange"],
        lw=1.5,
        drawstyle="steps-post",
        label="当前最优可行解",
        zorder=2,
    )
    ax0.axhline(float(baseline["total_mass"]), color=COLORS["black"], lw=0.8, ls=(0, (3, 2)))
    ax0.scatter(
        best["case_index"],
        best["total_mass"],
        marker="*",
        s=120,
        color=COLORS["orange"],
        edgecolors=COLORS["black"],
        linewidths=0.45,
        zorder=5,
    )
    annotate_best(
        ax0,
        best["case_index"],
        best["total_mass"],
        f"最优：{best['case_name']}\n质量 = {best['total_mass']:.4f}",
        (8, 12),
    )
    ax0.set_xlabel("已验证算例编号")
    ax0.set_ylabel("总质量")
    ax0.set_title("优化迭代过程")
    ax0.legend(loc="upper right", handlelength=1.6)
    beautify_axis(ax0)
    panel_label(ax0, "a")

    ax1.axhspan(pressure_limit, design["buckling_pressure"].max() * 1.08, color=COLORS["pale_green"], zorder=0)
    ax1.axhspan(0, pressure_limit, color=COLORS["pale_red"], zorder=0)
    ax1.axhline(
        pressure_limit,
        color=COLORS["black"],
        lw=0.85,
        ls=(0, (3, 2)),
        label=f"基准屈曲压力 = {pressure_limit:.5f}",
    )
    ax1.scatter(
        design.loc[~design["feasible"], "total_mass"],
        design.loc[~design["feasible"], "buckling_pressure"],
        s=18,
        facecolors="white",
        edgecolors=COLORS["gray"],
        linewidths=0.65,
        alpha=0.78,
    )
    sc1 = ax1.scatter(
        design.loc[design["feasible"], "total_mass"],
        design.loc[design["feasible"], "buckling_pressure"],
        c=design.loc[design["feasible"], "case_index"],
        cmap="viridis",
        s=28,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )
    ax1.scatter(
        best["total_mass"],
        best["buckling_pressure"],
        marker="*",
        s=125,
        color=COLORS["orange"],
        edgecolors=COLORS["black"],
        linewidths=0.45,
        zorder=5,
    )
    annotate_best(
        ax1,
        best["total_mass"],
        best["buckling_pressure"],
        f"减重 {100 * best['mass_reduction']:.1f}%\n屈曲压力 = {best['buckling_pressure']:.5f}",
        (18, 36),
    )
    cbar = fig.colorbar(sc1, ax=ax1, fraction=0.045, pad=0.02)
    cbar.set_label("算例编号")
    ax1.set_xlabel("总质量")
    ax1.set_ylabel("屈曲压力")
    ax1.set_title("质量-约束权衡")
    ax1.legend(loc="upper right", handlelength=1.6)
    beautify_axis(ax1)
    panel_label(ax1, "b")

    p_values = np.linspace(max(1.0, design["p"].min() * 0.9), design["p"].max() * 1.04, 300)
    n_limit = CYLINDER_LENGTH / p_values
    ax2.fill_between(
        p_values,
        n_limit,
        design["n"].max() + 0.8,
        color=COLORS["pale_red"],
        alpha=0.65,
        linewidth=0,
        label="n × p > 2650",
    )
    ax2.plot(p_values, n_limit, color=COLORS["black"], lw=0.9, ls=(0, (3, 2)), label="n × p = 2650")
    norm = Normalize(vmin=design["total_mass"].min(), vmax=design["total_mass"].max())
    sc2 = ax2.scatter(
        design["p"],
        design["n"],
        c=design["total_mass"],
        cmap="magma_r",
        norm=norm,
        s=np.where(design["feasible"], 31, 18),
        marker="o",
        edgecolors=np.where(design["feasible"], COLORS["black"], COLORS["gray"]),
        linewidths=np.where(design["feasible"], 0.35, 0.25),
        alpha=np.where(design["feasible"], 0.98, 0.55),
        zorder=3,
    )
    ax2.scatter(
        best["p"],
        best["n"],
        marker="*",
        s=125,
        color=COLORS["orange"],
        edgecolors=COLORS["black"],
        linewidths=0.45,
        zorder=5,
    )
    cbar2 = fig.colorbar(sc2, ax=ax2, fraction=0.045, pad=0.02)
    cbar2.set_label("总质量")
    ax2.set_xlabel("波距 p")
    ax2.set_ylabel("波纹数 n")
    ax2.set_title("设计空间可行域")
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend(loc="upper right", handlelength=1.6)
    beautify_axis(ax2)
    panel_label(ax2, "c")

    stages = ["Baseline", "LHS screening", "GPR-guided verification"]
    values = [float(baseline["total_mass"])]
    labels = ["unoptimized"]
    for stage in stages[1:]:
        subset = design[(design["stage"] == stage) & (design["feasible"])]
        if subset.empty:
            values.append(np.nan)
            labels.append("none")
        else:
            row = subset.sort_values("total_mass").iloc[0]
            values.append(float(row["total_mass"]))
            labels.append(str(row["case_name"]))

    x = np.arange(len(stages))
    bar_colors = [COLORS["gray"], COLORS["blue"], COLORS["orange"]]
    ax3.bar(x, values, color=bar_colors, width=0.58, edgecolor=COLORS["black"], linewidth=0.45)
    ax3.set_xticks(x, ["基准", "初始\nLHS", "GPR 引导\n验证"])
    ax3.set_ylabel("最优可行质量")
    ax3.set_title("各阶段最优已验证方案")
    for idx, value in enumerate(values):
        if np.isfinite(value):
            reduction = 1.0 - value / float(baseline["total_mass"])
            ax3.text(
                idx,
                value + 0.006,
                f"{value:.3f}\n({reduction:.0%})",
                ha="center",
                va="bottom",
                fontsize=6.3,
                color=COLORS["black"],
            )
    ax3.text(
        0.03,
        0.96,
        (
            f"当前最优解\n"
            f"{best['case_name']}\n"
            f"t={best['t']:.3f}, n={int(best['n'])}, p={int(best['p'])}, a={int(best['a'])}"
        ),
        transform=ax3.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color=COLORS["black"],
        bbox={"boxstyle": "round,pad=0.26", "fc": "white", "ec": COLORS["light_gray"], "lw": 0.6},
    )
    ax3.set_ylim(0, max(values) * 1.18)
    beautify_axis(ax3, minor=False)
    panel_label(ax3, "d")

    save_figure(fig, "cylinder_optimization_summary")
    plt.close(fig)


def draw_before_after_comparison(baseline: pd.Series, best: pd.Series) -> None:
    """单独输出“优化前 vs 优化后”对比柱状图。"""
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.6), constrained_layout=True)

    reduction = 1.0 - float(best["total_mass"]) / float(baseline["total_mass"])
    pressure_gain = float(best["buckling_pressure"]) / float(baseline["buckling_pressure"]) - 1.0

    # 左图：质量对比（越低越好）
    axes[0].bar(
        ["基准", "优化后"],
        [float(baseline["total_mass"]), float(best["total_mass"])],
        color=[COLORS["gray"], COLORS["orange"]],
        edgecolor=COLORS["black"],
        linewidth=0.5,
        width=0.58,
    )
    axes[0].set_title("总质量对比")
    axes[0].set_ylabel("total_mass")
    axes[0].text(
        1,
        float(best["total_mass"]) + 0.005,
        f"减重 {100 * reduction:.1f}%",
        ha="center",
        va="bottom",
        fontsize=6.3,
    )
    beautify_axis(axes[0], minor=False)

    # 右图：屈曲压力对比（越高越好）
    axes[1].bar(
        ["基准", "优化后"],
        [float(baseline["buckling_pressure"]), float(best["buckling_pressure"])],
        color=[COLORS["blue"], COLORS["green"]],
        edgecolor=COLORS["black"],
        linewidth=0.5,
        width=0.58,
    )
    axes[1].set_title("屈曲压力对比")
    axes[1].set_ylabel("buckling_pressure")
    axes[1].text(
        1,
        float(best["buckling_pressure"]) + 0.01 * float(best["buckling_pressure"]),
        f"提升 {100 * pressure_gain:.1f}%",
        ha="center",
        va="bottom",
        fontsize=6.3,
    )
    beautify_axis(axes[1], minor=False)

    fig.suptitle(f"优化前后性能对比（最优解：{best['case_name']}）", y=1.02, fontsize=8.4, fontweight="bold")
    save_figure(fig, "cylinder_before_after_comparison")
    plt.close(fig)


def draw_mass_constraint_map(
    baseline: pd.Series,
    design: pd.DataFrame,
    best: pd.Series,
    pressure_limit: float,
) -> None:
    """质量-屈曲压力平面图：直观展示可行域和最优解位置。"""
    fig, ax = plt.subplots(figsize=(3.55, 2.65), constrained_layout=True)

    feasible = design[design["feasible"]]
    infeasible = design[~design["feasible"]]

    ax.axhspan(pressure_limit, design["buckling_pressure"].max() * 1.08, color=COLORS["pale_green"], zorder=0)
    ax.axhspan(0, pressure_limit, color=COLORS["pale_red"], zorder=0)
    ax.axhline(pressure_limit, color=COLORS["black"], lw=0.85, ls=(0, (3, 2)))
    ax.scatter(
        infeasible["total_mass"],
        infeasible["buckling_pressure"],
        s=18,
        facecolors="white",
        edgecolors=COLORS["gray"],
        linewidths=0.65,
        alpha=0.82,
        label="不可行",
    )
    ax.scatter(
        feasible["total_mass"],
        feasible["buckling_pressure"],
        s=27,
        color=COLORS["green"],
        edgecolors="white",
        linewidths=0.35,
        alpha=0.95,
        label="可行",
    )
    ax.scatter(
        float(baseline["total_mass"]),
        pressure_limit,
        marker="s",
        s=32,
        color=COLORS["blue"],
        edgecolors=COLORS["black"],
        linewidths=0.4,
        label="基准",
        zorder=4,
    )
    ax.scatter(
        best["total_mass"],
        best["buckling_pressure"],
        marker="*",
        s=130,
        color=COLORS["orange"],
        edgecolors=COLORS["black"],
        linewidths=0.45,
        label="最优解",
        zorder=5,
    )
    annotate_best(
        ax,
        best["total_mass"],
        best["buckling_pressure"],
        f"{best['case_name']}\n减重：{100 * best['mass_reduction']:.1f}%",
        (54, 58),
    )
    ax.set_xlabel("总质量")
    ax.set_ylabel("屈曲压力")
    ax.set_title("筒体优化验证结果")
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=COLORS["gray"], markersize=4.5, label="不可行"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["green"], markeredgecolor="white", markersize=4.8, label="可行"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["blue"], markeredgecolor=COLORS["black"], markersize=4.8, label="基准"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=COLORS["orange"], markeredgecolor=COLORS["black"], markersize=7.5, label="最优解"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    beautify_axis(ax)

    save_figure(fig, "cylinder_mass_constraint_map")
    plt.close(fig)


def draw_parameter_response_grid(
    design: pd.DataFrame,
    best: pd.Series,
    pressure_limit: float,
) -> None:
    """核心关系图：t/n/p/a 与 mass/buckling_pressure 的二维关联。"""
    fig, axes = plt.subplots(2, 4, figsize=(7.25, 3.65), constrained_layout=True)

    infeasible = design[~design["feasible"]]
    feasible = design[design["feasible"]]

    for col_idx, param in enumerate(PARAMETER_COLS):
        for row_idx, response in enumerate(RESPONSE_COLS):
            ax = axes[row_idx, col_idx]

            ax.scatter(
                infeasible[param],
                infeasible[response],
                s=16,
                facecolors="white",
                edgecolors=COLORS["gray"],
                linewidths=0.55,
                alpha=0.72,
                zorder=2,
            )
            ax.scatter(
                feasible[param],
                feasible[response],
                s=24,
                color=COLORS["green"],
                edgecolors="white",
                linewidths=0.35,
                alpha=0.94,
                zorder=3,
            )

            gpr_round = design[design["stage"] == "GPR-guided verification"]
            ax.scatter(
                gpr_round[param],
                gpr_round[response],
                s=32,
                facecolors="none",
                edgecolors=COLORS["orange"],
                linewidths=0.55,
                alpha=0.72,
                zorder=4,
            )

            x_all, y_all = binned_median_line(design[param], design[response])
            if len(x_all):
                ax.plot(x_all, y_all, color=COLORS["gray"], lw=0.9, ls=(0, (2, 2)), zorder=1)

            x_feas, y_feas = binned_median_line(feasible[param], feasible[response])
            if len(x_feas):
                ax.plot(x_feas, y_feas, color=COLORS["green"], lw=1.2, zorder=5)

            ax.scatter(
                best[param],
                best[response],
                marker="*",
                s=95,
                color=COLORS["orange"],
                edgecolors=COLORS["black"],
                linewidths=0.4,
                zorder=6,
            )

            if response == "buckling_pressure":
                ax.axhline(pressure_limit, color=COLORS["black"], lw=0.75, ls=(0, (3, 2)), zorder=0)

            if row_idx == 1:
                ax.set_xlabel(PARAMETER_LABELS[param])
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

            if col_idx == 0:
                ax.set_ylabel(RESPONSE_LABELS[response])
            else:
                ax.set_ylabel("")

            if param in {"n", "p", "a"}:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

            beautify_axis(ax)

    axes[0, 0].set_title("参数-响应关系图", loc="left", fontsize=8.2, fontweight="bold")
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=COLORS["gray"], markersize=4.2, label="不可行"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["green"], markeredgecolor="white", markersize=4.8, label="可行"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=COLORS["orange"], markersize=5.0, label="GPR 验证轮"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=COLORS["orange"], markeredgecolor=COLORS["black"], markersize=7.5, label="最优解"),
        Line2D([0], [0], color=COLORS["green"], lw=1.2, label="可行点中位趋势"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 1.04),
        ncol=5,
        handlelength=1.5,
        columnspacing=1.2,
    )
    save_figure(fig, "cylinder_parameter_response_grid")
    plt.close(fig)


def draw_parallel_coordinates(design: pd.DataFrame, best: pd.Series) -> None:
    """平行坐标图：展示高维参数组合及最优解轨迹。"""
    variables = PARAMETER_COLS + RESPONSE_COLS
    display_labels = [
        "t",
        "n",
        "p",
        "a",
        "质量\n越低越好",
        "屈曲压力\n越高越好",
    ]
    normalized = design[variables].apply(normalize_to_unit)
    x = np.arange(len(variables))

    fig, ax = plt.subplots(figsize=(6.85, 2.95), constrained_layout=True)

    for _, row in normalized.loc[~design["feasible"]].iterrows():
        ax.plot(x, row.to_numpy(dtype=float), color=COLORS["gray"], lw=0.55, alpha=0.25, zorder=1)

    for stage, color, alpha, label in [
        ("LHS screening", COLORS["blue"], 0.34, "Feasible LHS"),
        ("GPR-guided verification", COLORS["green"], 0.45, "Feasible GPR-guided"),
    ]:
        subset = design[(design["feasible"]) & (design["stage"] == stage)]
        for idx in subset.index:
            ax.plot(x, normalized.loc[idx].to_numpy(dtype=float), color=color, lw=0.95, alpha=alpha, zorder=2)

    best_values = []
    for variable in variables:
        low = design[variable].min()
        high = design[variable].max()
        if abs(high - low) < 1e-12:
            best_values.append(0.0)
        else:
            best_values.append((best[variable] - low) / (high - low))

    ax.plot(
        x,
        best_values,
        color=COLORS["orange"],
        lw=2.4,
        marker="o",
        markersize=4.6,
        markeredgecolor=COLORS["black"],
        markeredgewidth=0.35,
        label=f"最优解：{best['case_name']}",
        zorder=5,
    )

    for xpos in x:
        ax.axvline(xpos, color=COLORS["light_gray"], lw=0.65, zorder=0)

    for xpos, variable in zip(x, variables):
        low = design[variable].min()
        high = design[variable].max()
        ax.text(xpos, 1.055, f"{high:.3g}", ha="center", va="bottom", fontsize=5.8, color=COLORS["gray"])
        ax.text(xpos, -0.075, f"{low:.3g}", ha="center", va="top", fontsize=5.8, color=COLORS["gray"])

    ax.set_xlim(x[0] - 0.15, x[-1] + 0.15)
    ax.set_ylim(-0.11, 1.11)
    ax.set_xticks(x, display_labels)
    ax.set_ylabel("归一化取值")
    ax.set_title("设计变量与响应的平行坐标图")
    legend_handles = [
        Line2D([0], [0], color=COLORS["gray"], lw=0.9, alpha=0.45, label="不可行"),
        Line2D([0], [0], color=COLORS["blue"], lw=1.2, alpha=0.75, label="LHS 可行点"),
        Line2D([0], [0], color=COLORS["green"], lw=1.2, alpha=0.75, label="GPR 引导可行点"),
        Line2D([0], [0], color=COLORS["orange"], marker="o", markerfacecolor=COLORS["orange"], markeredgecolor=COLORS["black"], lw=2.4, label="最优解"),
    ]
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.01, 0.5))
    beautify_axis(ax, minor=False)
    save_figure(fig, "cylinder_parallel_coordinates")
    plt.close(fig)


def draw_parameter_evolution_heatmap(design: pd.DataFrame, best: pd.Series) -> None:
    """演化热图：观察参数与响应随迭代过程的变化。"""
    ordered = design.sort_values("case_index").reset_index(drop=True)
    variables = ["t", "n", "p", "a", "n_times_p", "total_mass", "buckling_pressure"]
    labels = ["t", "n", "p", "a", "n × p", "质量", "屈曲压力"]
    matrix = ordered[variables].apply(normalize_to_unit).T.to_numpy(dtype=float)

    fig = plt.figure(figsize=(7.2, 3.25), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[0.18, 1.0])
    ax_strip = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0], sharex=ax_strip)

    stage_strip = np.where(ordered["stage"] == "LHS screening", 0.35, 0.80)[None, :]
    ax_strip.imshow(stage_strip, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    feasible_x = np.where(ordered["feasible"].to_numpy())[0]
    infeasible_x = np.where(~ordered["feasible"].to_numpy())[0]
    ax_strip.scatter(infeasible_x, np.zeros_like(infeasible_x), s=8, color=COLORS["gray"], alpha=0.45)
    ax_strip.scatter(feasible_x, np.zeros_like(feasible_x), s=10, color=COLORS["green"], alpha=0.85)
    best_pos = int(np.where(ordered["case_name"].to_numpy() == best["case_name"])[0][0])
    ax_strip.scatter([best_pos], [0], marker="*", s=75, color=COLORS["orange"], edgecolors=COLORS["black"], linewidths=0.35, zorder=4)
    ax_strip.axvline(INITIAL_LHS_LAST_CASE - 0.5, color=COLORS["black"], lw=0.8, ls=(0, (3, 2)))
    ax_strip.text(
        0.24,
        0.5,
        "LHS 初始采样",
        transform=ax_strip.transAxes,
        ha="center",
        va="center",
        fontsize=6.4,
        color=COLORS["black"],
    )
    ax_strip.text(
        0.74,
        0.5,
        "GPR 引导验证",
        transform=ax_strip.transAxes,
        ha="center",
        va="center",
        fontsize=6.4,
        color="white",
    )
    ax_strip.set_yticks([])
    ax_strip.set_ylabel("阶段", rotation=0, ha="right", va="center", fontsize=6.4)
    ax_strip.tick_params(axis="x", labelbottom=False, bottom=False, top=False)
    for spine in ax_strip.spines.values():
        spine.set_visible(False)

    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax.axvline(INITIAL_LHS_LAST_CASE - 0.5, color="white", lw=1.0, ls=(0, (3, 2)))
    ax.scatter([best_pos], [variables.index("total_mass")], marker="*", s=90, color=COLORS["orange"], edgecolors=COLORS["black"], linewidths=0.35, zorder=4)
    ax.scatter([best_pos], [variables.index("buckling_pressure")], marker="*", s=90, color=COLORS["orange"], edgecolors=COLORS["black"], linewidths=0.35, zorder=4)

    tick_positions = np.linspace(0, len(ordered) - 1, 7, dtype=int)
    ax.set_xticks(tick_positions, ordered.loc[tick_positions, "case_index"].astype(int))
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("已验证算例编号")
    ax.set_title("变量与响应随优化过程的演化", pad=6)
    ax.tick_params(top=False, right=False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.012)
    cbar.set_label("归一化取值")
    save_figure(fig, "cylinder_parameter_evolution_heatmap")
    plt.close(fig)


def draw_sampling_scatter_matrix(design: pd.DataFrame, best: pd.Series) -> None:
    """采样散点矩阵：对比初始 LHS、GPR 引导点与随机参考分布。"""
    rng = np.random.default_rng(GPR_RANDOM_SEED + 3)
    lhs = design[design["stage"] == "LHS screening"]
    gpr = design[design["stage"] == "GPR-guided verification"]
    random_ref = random_design_samples(design, len(lhs), rng)

    n_vars = len(PARAMETER_COLS)
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(6.6, 6.3), constrained_layout=True)

    for row_idx, y_col in enumerate(PARAMETER_COLS):
        for col_idx, x_col in enumerate(PARAMETER_COLS):
            ax = axes[row_idx, col_idx]

            if row_idx == col_idx:
                bins = 8 if x_col != "n" else np.arange(design[x_col].min() - 0.5, design[x_col].max() + 1.5, 1)
                ax.hist(
                    random_ref[x_col],
                    bins=bins,
                    color=COLORS["light_gray"],
                    edgecolor="white",
                    alpha=0.85,
                    density=True,
                    label="随机参考",
                )
                ax.hist(
                    lhs[x_col],
                    bins=bins,
                    histtype="step",
                    color=COLORS["blue"],
                    lw=1.1,
                    density=True,
                    label="初始 LHS",
                )
                ax.axvline(best[x_col], color=COLORS["orange"], lw=1.3)
            else:
                ax.scatter(
                    random_ref[x_col],
                    random_ref[y_col],
                    s=8,
                    color=COLORS["light_gray"],
                    alpha=0.55,
                    linewidths=0,
                )
                ax.scatter(
                    lhs[x_col],
                    lhs[y_col],
                    s=14,
                    color=COLORS["blue"],
                    edgecolors="white",
                    linewidths=0.25,
                    alpha=0.85,
                )
                ax.scatter(
                    gpr[x_col],
                    gpr[y_col],
                    s=16,
                    facecolors="none",
                    edgecolors=COLORS["orange"],
                    linewidths=0.65,
                    alpha=0.88,
                )
                ax.scatter(
                    best[x_col],
                    best[y_col],
                    marker="*",
                    s=75,
                    color=COLORS["orange"],
                    edgecolors=COLORS["black"],
                    linewidths=0.35,
                    zorder=4,
                )

            if row_idx == n_vars - 1:
                ax.set_xlabel(PARAMETER_LABELS[x_col])
            else:
                ax.set_xticklabels([])

            if col_idx == 0:
                ax.set_ylabel(PARAMETER_LABELS[y_col])
            else:
                ax.set_yticklabels([])

            if x_col in INTEGER_PARAMETER_COLS and row_idx != col_idx:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            if y_col in INTEGER_PARAMETER_COLS and row_idx != col_idx:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

            beautify_axis(ax, minor=False)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["light_gray"], markeredgecolor=COLORS["light_gray"], markersize=4.5, label="随机参考"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["blue"], markeredgecolor="white", markersize=4.8, label="初始 LHS"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=COLORS["orange"], markersize=5.0, label="GPR 引导验证"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=COLORS["orange"], markeredgecolor=COLORS["black"], markersize=7.5, label="最优解"),
    ]
    fig.suptitle("采样点在二维投影中的分布", y=1.02, fontsize=8.6, fontweight="bold")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.985), ncol=4, columnspacing=1.15)
    save_figure(fig, "cylinder_sampling_scatter_matrix")
    plt.close(fig)


def draw_gpr_validation(validation: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """预测值 vs 真实值：带置信区间并标注 R2/RMSE/MAE。"""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), constrained_layout=True)

    for ax, target in zip(axes, RESPONSE_COLS):
        subset = validation[validation["target"] == target].copy()
        target_metrics = metrics[metrics["target"] == target].iloc[0]
        x = subset["actual"].to_numpy(dtype=float)
        y = subset["predicted"].to_numpy(dtype=float)
        yerr = 1.96 * subset["std"].to_numpy(dtype=float)

        feasible = subset["feasible"].to_numpy(dtype=bool)
        ax.errorbar(
            x[~feasible],
            y[~feasible],
            yerr=yerr[~feasible],
            fmt="o",
            ms=3.0,
            mfc="white",
            mec=COLORS["gray"],
            ecolor=COLORS["light_gray"],
            elinewidth=0.55,
            capsize=1.6,
            alpha=0.82,
            label="不可行",
        )
        ax.errorbar(
            x[feasible],
            y[feasible],
            yerr=yerr[feasible],
            fmt="o",
            ms=3.2,
            mfc=COLORS["green"],
            mec="white",
            ecolor=COLORS["green"],
            elinewidth=0.55,
            capsize=1.6,
            alpha=0.88,
            label="可行",
        )

        low = min(np.nanmin(x), np.nanmin(y))
        high = max(np.nanmax(x), np.nanmax(y))
        pad = (high - low) * 0.08
        ax.plot([low - pad, high + pad], [low - pad, high + pad], color=COLORS["black"], lw=0.85, ls=(0, (3, 2)))
        ax.fill_between(
            [low - pad, high + pad],
            [low - pad - 0.1 * (high - low), high + pad - 0.1 * (high - low)],
            [low - pad + 0.1 * (high - low), high + pad + 0.1 * (high - low)],
            color=COLORS["light_gray"],
            alpha=0.25,
            linewidth=0,
        )
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"真实{RESPONSE_LABELS[target]}")
        ax.set_ylabel(f"预测{RESPONSE_LABELS[target]}")
        ax.set_title(RESPONSE_LABELS[target])
        ax.text(
            0.05,
            0.95,
            (
                f"五折交叉验证\n"
                f"$R^2$ = {target_metrics['r2']:.3f}\n"
                f"RMSE = {target_metrics['rmse']:.3g}\n"
                f"MAE = {target_metrics['mae']:.3g}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": COLORS["light_gray"], "lw": 0.6},
        )
        beautify_axis(ax)

    axes[0].legend(loc="lower right")
    fig.suptitle("高斯过程回归模型验证：预测值与真实值", y=1.03, fontsize=8.4, fontweight="bold")
    save_figure(fig, "cylinder_gpr_predicted_vs_actual")
    plt.close(fig)


def response_surface_parameters(sensitivity: pd.DataFrame) -> list[str]:
    """按敏感性自动挑选两个最重要参数用于响应面绘制。"""
    combined = (
        sensitivity.groupby("parameter", as_index=False)["normalized_gp_effect"]
        .mean()
        .sort_values("normalized_gp_effect", ascending=False)
    )
    selected = combined["parameter"].head(2).tolist()
    if len(selected) < 2:
        selected = ["t", "n"]
    return selected


def draw_gpr_response_surface(
    design: pd.DataFrame,
    best: pd.Series,
    models: dict[str, SimpleGaussianProcess],
    sensitivity: pd.DataFrame,
    pressure_limit: float,
) -> None:
    """二维响应面/等高线图：固定其余参数为最优解取值。"""
    x_col, y_col = response_surface_parameters(sensitivity)
    grid_n = 120
    x_values = np.linspace(design[x_col].min(), design[x_col].max(), grid_n)
    y_values = np.linspace(design[y_col].min(), design[y_col].max(), grid_n)
    xx, yy = np.meshgrid(x_values, y_values)

    surface_frame = pd.DataFrame({col: np.full(xx.size, best[col]) for col in PARAMETER_COLS})
    surface_frame[x_col] = xx.ravel()
    surface_frame[y_col] = yy.ravel()
    surface_frame = enforce_integer_parameters(surface_frame)
    valid_mask = physical_feasible(surface_frame)

    predictions = {}
    for target, model in models.items():
        pred, _ = model.predict(surface_frame[PARAMETER_COLS].to_numpy(dtype=float))
        pred[~valid_mask] = np.nan
        predictions[target] = pred.reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.95), constrained_layout=True)
    cmap_by_target = {"total_mass": "viridis_r", "buckling_pressure": "viridis"}

    for ax, target in zip(axes, RESPONSE_COLS):
        zz = predictions[target]
        levels = 16
        contour = ax.contourf(xx, yy, zz, levels=levels, cmap=cmap_by_target[target])
        ax.contour(xx, yy, zz, levels=8, colors="white", linewidths=0.35, alpha=0.75)
        if target == "buckling_pressure":
            ax.contour(xx, yy, zz, levels=[pressure_limit], colors=COLORS["black"], linewidths=1.0, linestyles="--")

        ax.scatter(
            design[x_col],
            design[y_col],
            s=12,
            facecolors="none",
            edgecolors=np.where(design["feasible"], COLORS["green"], COLORS["gray"]),
            linewidths=0.45,
            alpha=0.72,
        )
        ax.scatter(
            best[x_col],
            best[y_col],
            marker="*",
            s=130,
            color=COLORS["orange"],
            edgecolors=COLORS["black"],
            linewidths=0.45,
            zorder=5,
        )
        ax.set_xlabel(PARAMETER_LABELS[x_col])
        ax.set_ylabel(PARAMETER_LABELS[y_col])
        ax.set_title(f"高斯过程响应面：{RESPONSE_LABELS[target]}")
        if x_col in INTEGER_PARAMETER_COLS:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        if y_col in INTEGER_PARAMETER_COLS:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(RESPONSE_LABELS[target])
        beautify_axis(ax)

    fixed = ", ".join([f"{col}={best[col]:.3g}" for col in PARAMETER_COLS if col not in {x_col, y_col}])
    fig.text(
        0.5,
        -0.035,
        f"未显示变量固定为当前已验证最优解（{fixed}）；已屏蔽 n × p > 2650 的不可行网格。",
        ha="center",
        va="top",
        fontsize=6.4,
        color=COLORS["gray"],
    )
    save_figure(fig, "cylinder_gpr_response_surface")
    plt.close(fig)


def draw_gpr_sensitivity_bar(sensitivity: pd.DataFrame) -> None:
    """敏感性柱状图：展示置换效应与线性相关性。"""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.65), constrained_layout=True, sharey=True)

    for ax, target in zip(axes, RESPONSE_COLS):
        subset = sensitivity[sensitivity["target"] == target].sort_values("normalized_gp_effect", ascending=True)
        y = np.arange(len(subset))
        ax.barh(
            y,
            subset["normalized_gp_effect"],
            color=COLORS["blue"] if target == "total_mass" else COLORS["green"],
            edgecolor=COLORS["black"],
            linewidth=0.4,
            alpha=0.92,
            label="GPR 置换效应",
        )
        ax.scatter(
            subset["abs_pearson_correlation"],
            y,
            marker="D",
            s=18,
            color=COLORS["orange"],
            edgecolors=COLORS["black"],
            linewidths=0.3,
            label="|皮尔逊相关系数 r|",
            zorder=3,
        )
        ax.set_yticks(y, [PARAMETER_LABELS[p] for p in subset["parameter"]])
        ax.set_xlabel("归一化敏感性")
        ax.set_title(RESPONSE_LABELS[target])
        ax.set_xlim(0, max(1.0, subset[["normalized_gp_effect", "abs_pearson_correlation"]].max().max() * 1.08))
        beautify_axis(ax, minor=False)

    axes[1].legend(loc="lower right")
    fig.suptitle("设计参数敏感性分析", y=1.03, fontsize=8.4, fontweight="bold")
    save_figure(fig, "cylinder_gpr_sensitivity_bar")
    plt.close(fig)


def draw_gpr_sobol_bar(sobol_df: pd.DataFrame) -> None:
    """Sobol 敏感性柱状图：并排展示 S1 与 ST。"""
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8), constrained_layout=True, sharey=True)

    for ax, target in zip(axes, RESPONSE_COLS):
        subset = sobol_df[sobol_df["target"] == target].copy()
        subset = subset.set_index("parameter").reindex(PARAMETER_COLS).reset_index()
        y = np.arange(len(subset))
        h = 0.34

        ax.barh(
            y - h / 2,
            subset["sobol_s1"],
            height=h,
            color=COLORS["blue"],
            edgecolor=COLORS["black"],
            linewidth=0.4,
            alpha=0.9,
            label="Sobol 一阶指数 S1",
        )
        ax.barh(
            y + h / 2,
            subset["sobol_st"],
            height=h,
            color=COLORS["green"],
            edgecolor=COLORS["black"],
            linewidth=0.4,
            alpha=0.9,
            label="Sobol 总阶指数 ST",
        )

        ax.set_yticks(y, [PARAMETER_LABELS[p] for p in subset["parameter"]])
        ax.set_xlabel("Sobol 指数")
        ax.set_xlim(0, 1.0)
        ax.set_title(RESPONSE_LABELS[target])
        beautify_axis(ax, minor=False)

    axes[1].legend(loc="lower right")
    fig.suptitle("Sobol 全局敏感性分析（基于 GPR 代理模型）", y=1.03, fontsize=8.4, fontweight="bold")
    save_figure(fig, "cylinder_gpr_sobol_bar")
    plt.close(fig)


def export_ranked_table(design: pd.DataFrame) -> Path:
    """导出可行解按质量排序表，便于论文附录复核。"""
    cols = [
        "case_name",
        "t",
        "n",
        "p",
        "a",
        "n_times_p",
        "total_mass",
        "eigenvalue",
        "buckling_pressure",
        "mass_reduction",
        "buckling_margin",
    ]
    ranked = design[design["feasible"]].sort_values("total_mass")[cols].copy()
    out_path = OUT_DIR / "cylinder_feasible_ranked.csv"
    ranked.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    """主流程：读取数据、建模评估、出图、导出指标表。"""
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    _, baseline, design, best, pressure_limit = load_data()
    gpr_models = fit_gpr_models(design, RESPONSE_COLS)
    validation, metrics = cross_validated_gpr_predictions(design, RESPONSE_COLS)
    sensitivity = gp_permutation_sensitivity(design, gpr_models)
    sobol_df = sobol_sensitivity_from_gp(design, gpr_models)

    draw_summary_figure(baseline, design, best, pressure_limit)
    draw_before_after_comparison(baseline, best)
    draw_mass_constraint_map(baseline, design, best, pressure_limit)
    draw_parameter_response_grid(design, best, pressure_limit)
    draw_parallel_coordinates(design, best)
    draw_parameter_evolution_heatmap(design, best)
    draw_sampling_scatter_matrix(design, best)
    draw_gpr_validation(validation, metrics)
    draw_gpr_sensitivity_bar(sensitivity)
    draw_gpr_sobol_bar(sobol_df)
    draw_viz_tradeoff_figure()
    draw_gpr_response_surface(design, best, gpr_models, sensitivity, pressure_limit)
    ranked_path = export_ranked_table(design)
    metrics_path = OUT_DIR / "cylinder_gpr_validation_metrics.csv"
    validation_path = OUT_DIR / "cylinder_gpr_validation_predictions.csv"
    sensitivity_path = OUT_DIR / "cylinder_gpr_sensitivity.csv"
    sobol_path = OUT_DIR / "cylinder_gpr_sobol_indices.csv"
    metrics.to_csv(metrics_path, index=False)
    validation.to_csv(validation_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)
    sobol_df.to_csv(sobol_path, index=False)
    cleanup_temp_files()

    print(f"Style: {style_name}")
    print(f"Rows: {len(design)} verified designs")
    print(f"Feasible designs: {int(design['feasible'].sum())}")
    print(f"Baseline pressure limit: {pressure_limit:.5f}")
    print(f"Current optimum: {best['case_name']}")
    print(f"  total_mass = {best['total_mass']:.8g}")
    print(f"  buckling_pressure = {best['buckling_pressure']:.8g}")
    print(f"  mass_reduction = {100 * best['mass_reduction']:.2f}%")
    print(f"Saved figures to: {OUT_DIR}")
    print(f"Saved ranked feasible table to: {ranked_path}")
    print(f"Saved GPR validation metrics to: {metrics_path}")
    print(f"Saved GPR validation predictions to: {validation_path}")
    print(f"Saved GPR sensitivity table to: {sensitivity_path}")
    print(f"Saved Sobol sensitivity table to: {sobol_path}")


if __name__ == "__main__":
    main()
