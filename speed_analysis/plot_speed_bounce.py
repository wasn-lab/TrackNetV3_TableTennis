'''
1. 慣用手斜線
2. 慣用手直線
3. 非慣用手斜線
4. 非慣用手直線
5. 慣用手、非慣用手 斜線
6. 慣用手、非慣用手 直線

左右手
'''
# python speed_analysis/plot_speed_bounce.py --input /path/to/your_stroke_zone.csv
# python speed_analysis/plot_speed_bounce.py --input pred_result_NO7
# python speed_analysis/plot_speed_bounce.py --input C0083_stroke_zone.csv --target_mode r12

import argparse
import re
from pathlib import Path
from typing import Optional, List

import matplotlib.pyplot as plt
import pandas as pd


def _series_to_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes", "y"])


def get_row_from_zone(zone_label):
    """
    Example:
    C8R2 -> 2
    C7R1 -> 1
    """
    if pd.isna(zone_label):
        return None

    m = re.search(r"R(\d+)", str(zone_label))
    if not m:
        return None

    return int(m.group(1))


def is_correct_target(row_num, landing_order, target_mode):
    """
    landing_order 從 1 開始，只計算有落點的球。
    """
    if row_num is None:
        return False

    if target_mode == "none":
        return False

    if target_mode == "r12":
        return row_num in [1, 2]

    if target_mode == "r34":
        return row_num in [3, 4]

    if target_mode == "r12_r34":
        if landing_order % 2 == 1:
            return row_num in [1, 2]
        else:
            return row_num in [3, 4]

    if target_mode == "r34_r12":
        if landing_order % 2 == 1:
            return row_num in [3, 4]
        else:
            return row_num in [1, 2]

    return False


def plot_one_csv(
    csv_path: Path,
    speed_col: str,
    target_mode: str = "none",
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    df = pd.read_csv(csv_path)

    required_cols = ["stroke_id", speed_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[SKIP] {csv_path} missing columns: {missing}")
        return None

    plot_cols = ["stroke_id", speed_col]

    if "zone_label" in df.columns:
        plot_cols.append("zone_label")

    if "bounce_frame" in df.columns:
        plot_cols.append("bounce_frame")

    if "in_table" in df.columns:
        plot_cols.append("in_table")

    plot_df = df[plot_cols].copy()

    plot_df["stroke_id"] = pd.to_numeric(plot_df["stroke_id"], errors="coerce")
    plot_df[speed_col] = pd.to_numeric(plot_df[speed_col], errors="coerce")

    if "bounce_frame" in plot_df.columns:
        plot_df["bounce_frame"] = pd.to_numeric(plot_df["bounce_frame"], errors="coerce").fillna(0)

    plot_df = plot_df.dropna(subset=["stroke_id", speed_col]).sort_values("stroke_id")

    if plot_df.empty:
        print(f"[SKIP] {csv_path} has no valid {speed_col} data")
        return None

    plot_df["stroke_id"] = plot_df["stroke_id"].astype(int)

    plot_df["has_landing"] = False
    plot_df["correct_target"] = False

    if "zone_label" in plot_df.columns:
        plot_df["row_num"] = plot_df["zone_label"].apply(get_row_from_zone)

        # 有 zone_label 就視為有落點
        plot_df["has_landing"] = plot_df["row_num"].notna()

        # 如果有 bounce_frame，就再要求 bounce_frame > 0
        if "bounce_frame" in plot_df.columns:
            plot_df["has_landing"] = plot_df["has_landing"] & (plot_df["bounce_frame"] > 0)

        # 如果有 in_table，就再要求 in_table == True
        if "in_table" in plot_df.columns:
            plot_df["has_landing"] = plot_df["has_landing"] & _series_to_bool(plot_df["in_table"])

        correct_list = []

        for _, row in plot_df.iterrows():
            if row["has_landing"]:
                stroke_id = int(row["stroke_id"])

                correct = is_correct_target(
                    row_num=row["row_num"],
                    landing_order=stroke_id,
                    target_mode=target_mode,
                )
                correct_list.append(correct)
            else:
                correct_list.append(False)

        plot_df["correct_target"] = correct_list

        # 有進桌上但沒有打到目標區：綠色
        plot_df["wrong_landing"] = plot_df["has_landing"] & (~plot_df["correct_target"])

        # 統計只看 stroke 2~26
        eval_df = plot_df[
            (plot_df["stroke_id"] >= 2) &
            (plot_df["stroke_id"] <= 26)
        ].copy()

        eval_total = len(eval_df)
        eval_correct = int(eval_df["correct_target"].sum())
        eval_in_table = int(eval_df["has_landing"].sum())
        eval_wrong_landing = int(eval_df["wrong_landing"].sum())

        accuracy = eval_correct / eval_total * 100 if eval_total > 0 else 0
        in_rate = eval_in_table / eval_total * 100 if eval_total > 0 else 0
    else:
        eval_df = plot_df[
            (plot_df["stroke_id"] >= 2) &
            (plot_df["stroke_id"] <= 26)
        ].copy()
        eval_total = len(eval_df)
        eval_correct = 0
        eval_in_table = 0
        eval_wrong_landing = 0
        accuracy = 0
        in_rate = 0

    # Mean / Max / Median 只用紅點，也就是 eval strokes 2~26 裡 correct_target=True 的球。
    # 這樣藍點不會進入 max，右邊統計框會和紅點一致。
    stat_df = eval_df[eval_df["correct_target"]].copy()
    if stat_df.empty:
        mean_val = float("nan")
        max_val = float("nan")
        median_val = float("nan")
    else:
        mean_val = stat_df[speed_col].mean()
        max_val = stat_df[speed_col].max()
        median_val = stat_df[speed_col].median()

    correct_count = int(plot_df["correct_target"].sum())
    landing_count = int(plot_df["has_landing"].sum())

    save_dir = csv_path.parent if out_dir is None else Path(out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    safe_speed_col = speed_col.replace("/", "_").replace("\\", "_").replace(" ", "_")
    out_path = save_dir / f"{csv_path.stem}_{safe_speed_col}_bounce_line.png"

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[5, 1])

    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])

    # 原本的藍色折線
    ax.plot(
        plot_df["stroke_id"],
        plot_df[speed_col],
        marker="o",
        label=speed_col,
    )

    # 正確落點用紅色標出
    correct_df = plot_df[plot_df["correct_target"]].copy()
    if not correct_df.empty:
        ax.scatter(
            correct_df["stroke_id"],
            correct_df[speed_col],
            color="red",
            s=80,
            zorder=5,
            label="correct landing target",
        )

    # 有進桌上但不正確：綠色
    wrong_df = plot_df[plot_df["wrong_landing"]].copy()
    if not wrong_df.empty:
        ax.scatter(
            wrong_df["stroke_id"],
            wrong_df[speed_col],
            color="#1ed14b",
            s=80,
            zorder=5,
            label="in table but wrong target",
        )

    ax.set_ylim(bottom=0)

    for _, row in plot_df.iterrows():
        x = int(row["stroke_id"])
        y = float(row[speed_col])

        if bool(row["correct_target"]):
            text_color = "red"
        elif bool(row["wrong_landing"]):
            text_color = "green"
        else:
            text_color = "black"

        ax.annotate(
            f"{y:.1f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color=text_color,
        )

    ax.set_title(f"{speed_col}\n{csv_path.name}")
    ax.set_xlabel("Stroke ID")
    ax.set_ylabel("Speed (km/h)")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(plot_df["stroke_id"].tolist())
    ax.legend()

    stats_text = (
        f"Mean: {mean_val:.2f} km/h\n"
        f"Max: {max_val:.2f} km/h\n"
        f"Median: {median_val:.2f} km/h\n"
        f"Stat source: correct red points\n"
        f"Target: {target_mode}\n"
        f"Eval strokes: 2-26\n"
        f"Total: {eval_total}\n"
        f"In table: {eval_in_table}\n"
        f"Correct: {eval_correct}\n"
        f"Wrong target: {eval_wrong_landing}\n"
        f"In rate: {in_rate:.1f}%\n"
        f"Accuracy: {accuracy:.1f}%"
    )

    ax_info.axis("off")
    ax_info.text(
        0.05,
        0.95,
        stats_text,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[OK] {csv_path} -> {out_path}")
    return out_path


def collect_csv_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        return sorted(input_path.rglob("*_stroke_zone.csv"))

    raise FileNotFoundError(f"Input path not found: {input_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot a selected speed column for one CSV or all *_stroke_zone.csv files under a folder."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to one *_stroke_zone.csv file or a root folder.",
    )
    parser.add_argument(
        "--speed",
        type=str,
        default="net_zone_max_speed_kmh",
        help="Speed column name to plot. Example: net_zone_max_speed_kmh / max_speed_kmh / avg_speed_kmh",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        default="none",
        choices=["none", "r12", "r34", "r12_r34", "r34_r12"],
        help=(
            "Target mode: "
            "r12 = R1/R2 correct, "
            "r34 = R3/R4 correct, "
            "r12_r34 = odd landing R1/R2, even landing R3/R4, "
            "r34_r12 = odd landing R3/R4, even landing R1/R2."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Optional output folder. Default: save next to each CSV.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    speed_col = args.speed
    out_dir = args.out_dir

    csv_files = collect_csv_files(input_path)
    if not csv_files:
        print(f"[INFO] No *_stroke_zone.csv found under: {input_path}")
        return

    print(f"[INFO] Found {len(csv_files)} csv files")
    print(f"[INFO] Plot speed column: {speed_col}")
    print(f"[INFO] Target mode: {args.target_mode}")

    ok_count = 0
    skip_count = 0

    for csv_path in csv_files:
        result = plot_one_csv(
            csv_path,
            speed_col=speed_col,
            target_mode=args.target_mode,
            out_dir=out_dir,
        )
        if result is None:
            skip_count += 1
        else:
            ok_count += 1

    print("=" * 60)
    print(f"[DONE] success={ok_count}, skipped={skip_count}, total={len(csv_files)}")


if __name__ == "__main__":
    main()