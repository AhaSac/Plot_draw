"""
plot_08_gpr_predicted_vs_actual.py

输出 PNG：figures/head/高斯过程回归模型验证：预测值与真实值.png
功能：GPR 预测值-真实值验证图：展示五折交叉验证误差和置信区间。
"""

from head_plot_common import *


def is_mass_target(target: str, label: str) -> bool:
    """判断当前响应量是否为质量；用于把质量从 t 换算为 kg（×1000）。"""
    target_text = str(target).lower()
    label_text = str(label).lower()
    mass_keywords = ("mass", "weight", "质量", "重量")
    return any(k in target_text for k in mass_keywords) or any(k in label_text for k in mass_keywords)


# ============================== 对应 PNG：高斯过程回归模型验证：预测值与真实值.png ==============================
# 功能：GPR 预测值-真实值验证图：展示五折交叉验证误差和置信区间。
def draw_gpr_validation(validation: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """预测值 vs 真实值：带置信区间并标注 R2/RMSE/MAE。"""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), constrained_layout=True)

    for ax, target in zip(axes, RESPONSE_COLS):
        subset = validation[validation["target"] == target].copy()
        target_metrics = metrics[metrics["target"] == target].iloc[0]

        label = RESPONSE_LABELS[target]
        scale = 1.0
        unit =  ""

        x = subset["actual"].to_numpy(dtype=float) * scale
        y = subset["predicted"].to_numpy(dtype=float) * scale
        yerr = 1.96 * subset["std"].to_numpy(dtype=float) * scale

        feasible = subset["feasible"].to_numpy(dtype=bool)

        ax.errorbar(
            x[~feasible],
            y[~feasible],
            yerr=yerr[~feasible],
            fmt="o",
            ms=2.7,
            mfc="white",
            mec=COLORS["gray"],
            mew=0.3,
            ecolor=COLORS["light_gray"],
            elinewidth=0.5,
            capsize=1.4,
            alpha=0.82,
            label="不可行",
        )

        ax.errorbar(
            x[feasible],
            y[feasible],
            yerr=yerr[feasible],
            fmt="o",
            ms=2.8,
            mfc=COLORS["green"],
            mec="white",
            mew=0.1,
            ecolor=COLORS["green"],
            elinewidth=0.5,
            capsize=1.4,
            alpha=0.88,
            label="可行",
        )

        low = min(np.nanmin(x), np.nanmin(y))
        high = max(np.nanmax(x), np.nanmax(y))
        span = high - low if high > low else 1.0
        pad = span * 0.08
        band = 0.10 * span

        ax.plot([low - pad, high + pad], [low - pad, high + pad], color=COLORS["black"], lw=0.85, ls=(0, (3, 2)))
        ax.fill_between(
            [low - pad, high + pad],
            [low - pad - band, high + pad - band],
            [low - pad + band, high + pad + band],
            color=COLORS["light_gray"],
            alpha=0.25,
            linewidth=0,
        )

        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"真实{label}{unit}")
        ax.set_ylabel(f"预测{label}{unit}")

        rmse_disp = float(target_metrics["rmse"]) * scale
        mae_disp = float(target_metrics["mae"]) * scale

        ax.text(
            0.05,
            0.95,
            (
                f"五折交叉验证\n"
                f"$R^2$ = {target_metrics['r2']:.3f}\n"
                f"RMSE = {rmse_disp:.3g}{unit}\n"
                f"MAE = {mae_disp:.3g}{unit}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": COLORS["light_gray"], "lw": 0.6},
        )

        beautify_axis(ax)

    axes[0].legend(loc="lower right")
    save_figure(fig, "高斯过程回归模型验证：预测值与真实值")
    plt.close(fig)


def main() -> None:
    """单独运行本文件时，只生成对应的 1 张 PNG。"""
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    _, baseline, design, best, pressure_limit = load_data()
    validation, metrics = cross_validated_gpr_predictions(design, RESPONSE_COLS)
    draw_gpr_validation(validation, metrics)
    cleanup_temp_files()
    print(f"Saved: {OUT_DIR / '高斯过程回归模型验证：预测值与真实值.png'}")
    print(f"Style: {style_name}")


if __name__ == "__main__":
    main()
