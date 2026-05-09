"""
筒体论文图绘制公共模块。

功能：
1. 统一导入包、路径、颜色、字体、参数名等全局配置；
2. 读取 cylinder.csv，并生成 feasibility/stage/mass_reduction 等派生字段；
3. 提供所有绘图脚本共享的坐标轴美化、标注、保存、GPR 评估和敏感性分析函数。

使用方式：
- 不建议单独运行本文件；
- 每张 PNG 对应一个 plot_*.py 脚本；
- 批量出图运行 cylinder_figures_split_by_png.py。

运行前请确认本目录中存在：
- cylinder.csv（用于 01-11 主图；12-14 的 viz 图使用脚本内置 VIZ_CASE_DATA）
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


# -----------------------------------------------------------------------------
# 自包含 GPR 设置
# -----------------------------------------------------------------------------
# 原始脚本曾从 cylinder_gpr_optimization.py 中导入以下 3 个对象：
#     LENGTH_SCALE_GRID, NOISE, SimpleGaussianProcess
# 为避免额外依赖，这里直接内置一个轻量级 RBF 高斯过程模型。
# 后续如果只想改画图，不需要碰这里；如果想改变 GPR 平滑程度，可以调整
# LENGTH_SCALE_GRID 或 NOISE。
LENGTH_SCALE_GRID = np.array([0.35, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 5.00], dtype=float)
NOISE = 1.0e-6


class SimpleGaussianProcess:
    """轻量级 RBF 高斯过程回归模型，用于论文图中的代理模型展示。

    设计目标：
    - 不依赖 scikit-learn；
    - 不再需要 cylinder_gpr_optimization.py；
    - 输入和输出会自动标准化，适合 t/n/p/a 量纲差异较大的数据。
    """

    def __init__(self, length_scale_grid: np.ndarray | list[float] = LENGTH_SCALE_GRID, noise: float = NOISE) -> None:
        self.length_scale_grid = np.asarray(length_scale_grid, dtype=float)
        self.noise = float(noise)
        self.length_scale_: float | None = None
        self.x_mean_: np.ndarray | None = None
        self.x_std_: np.ndarray | None = None
        self.y_mean_: float | None = None
        self.y_std_: float | None = None
        self.x_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.lower_cholesky_: np.ndarray | None = None
        self.alpha_: np.ndarray | None = None

    @staticmethod
    def _safe_std(values: np.ndarray) -> np.ndarray:
        """避免标准差为 0 导致除零。"""
        std = values.std(axis=0, ddof=0)
        return np.where(std < 1.0e-12, 1.0, std)

    @staticmethod
    def _rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float) -> np.ndarray:
        """RBF 核函数。"""
        diff = x1[:, None, :] - x2[None, :, :]
        sq_dist = np.sum(diff * diff, axis=2)
        return np.exp(-0.5 * sq_dist / (length_scale**2))

    def _standardize_x(self, x: np.ndarray) -> np.ndarray:
        if self.x_mean_ is None or self.x_std_ is None:
            raise RuntimeError("SimpleGaussianProcess has not been fitted yet.")
        return (np.asarray(x, dtype=float) - self.x_mean_) / self.x_std_

    def _fit_with_length_scale(self, x_scaled: np.ndarray, y_scaled: np.ndarray, length_scale: float) -> tuple[float, np.ndarray, np.ndarray]:
        """给定 length_scale 拟合模型，并返回负对数边际似然。"""
        n = len(x_scaled)
        kernel = self._rbf_kernel(x_scaled, x_scaled, length_scale)
        kernel = kernel + (self.noise + 1.0e-10) * np.eye(n)

        # 小样本数据中可能出现数值病态，逐步增加 jitter 保证 Cholesky 分解稳定。
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
        """拟合 GPR 模型。"""
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
        self.y_train_ = y_scaled
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """预测均值和标准差。"""
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

def load_viz_dataframe() -> pd.DataFrame:
    """读取内置 VIZ_CASE_DATA，返回 viz 三张补充图共用的数据表。"""
    return pd.DataFrame(VIZ_CASE_DATA, columns=["name", "t", "n", "p", "a", "mass", "margin"])


def compute_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """按质量从小到大扫描，提取 mass-margin 权衡图中的 Pareto 前沿。"""
    pareto_rows = []
    sorted_df = df.sort_values("mass").reset_index(drop=True)
    max_margin = -np.inf
    for _, row in sorted_df.iterrows():
        if row["margin"] > max_margin:
            pareto_rows.append(row)
            max_margin = row["margin"]
    return pd.DataFrame(pareto_rows)
