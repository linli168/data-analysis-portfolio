# -*- coding: utf-8 -*-
"""
餐饮门店经营诊断看板
====================
基于北京市餐饮商铺数据（4.4w 门店真实数据）构建门店分层与经营风险看板：

  1. 门店健康分层（A 明星 / B 潜力 / C 口碑风险 / C 边缘 / D 评分缺失）
  2. 商圈经营诊断（C 类占比、评分缺失率）
  3. 菜系经营诊断（评分、人均消费、门店规模）
  4. 风险门店清单（脱敏：仅保留分析字段，不含店名/地址/电话）
  5. 输出：自包含 HTML 看板 + 图表 + Markdown 报告 + 风险清单 CSV

运行：
  python build_dashboard.py                    # 自动在上级目录查找真实数据
  python build_dashboard.py --data 数据文件.xlsx

说明：本脚本读取真实商铺数据进行分析；原始数据不随仓库公开，
      仓库内仅提供 sample_store_data.csv 展示表结构。
"""

import base64
import os
import sys

import matplotlib

# 字体缓存放到项目内可写目录
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
plt.rcParams["axes.unicode_minus"] = False
try:
    from matplotlib import font_manager

    for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
except Exception:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
DEFAULT_DATA = os.path.join(SCRIPT_DIR, "..", "01-restaurant-data-processing", "北京市餐饮商铺数据(4.4w).xlsx")
RATING_HIGH = 4.5  # 高评分门槛


def find_data_path() -> str:
    """查找真实数据文件：优先 --data 参数，其次上级目录，再当前目录"""
    if "--data" in sys.argv:
        idx = sys.argv.index("--data")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    for p in [DEFAULT_DATA, "北京市餐饮商铺数据(4.4w).xlsx",
              os.path.join(SCRIPT_DIR, "..", "北京市餐饮商铺数据(4.4w).xlsx")]:
        if os.path.exists(p):
            return p
    return DEFAULT_DATA


def classify_tier(df: pd.DataFrame, comment_median: float) -> pd.Series:
    """门店健康分层（基于评分与评论量两个可观测维度）"""
    has_rating = df["评分"].notna()
    rating_high = df["评分"] >= RATING_HIGH
    comment_high = df["评论数量"] >= comment_median

    tier = pd.Series("D_评分缺失", index=df.index, dtype=object)
    tier[has_rating & rating_high & comment_high] = "A_明星店"
    tier[has_rating & rating_high & ~comment_high] = "B_潜力店"
    tier[has_rating & ~rating_high & comment_high] = "C_口碑风险"
    tier[has_rating & ~rating_high & ~comment_high] = "C_边缘店"
    return tier


def load_data() -> pd.DataFrame:
    path = find_data_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到真实数据文件：{path}\n"
            "请将 北京市餐饮商铺数据(4.4w).xlsx 放入项目根目录，"
            "或用 --data 指定文件路径。"
        )
    df = pd.read_excel(path)
    print(f"已加载真实数据：{path}（{df.shape[0]} 行 × {df.shape[1]} 列）")
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data()

    # ---------- 门店分层 ----------
    comment_median = float(df[df["评分"].notna()]["评论数量"].median())
    print(f"分层阈值：高评分 >= {RATING_HIGH}，高评论 >= 中位数 {comment_median:.0f} 条")
    df["门店分层"] = classify_tier(df, comment_median)
    rated = df[df["评分"].notna()]

    tier_order = ["A_明星店", "B_潜力店", "C_口碑风险", "C_边缘店", "D_评分缺失"]
    tier_stats = df["门店分层"].value_counts().reindex(tier_order).fillna(0).astype(int)

    c_count = int(tier_stats[["C_口碑风险", "C_边缘店"]].sum())
    rated_ratio = len(rated) / len(df)

    # ---------- KPI ----------
    kpi = {
        "总门店数": f"{len(df):,}",
        "有效评分率": f"{rated_ratio*100:.1f}%",
        "平均评分(有评分)": f"{rated['评分'].mean():.2f}",
        "人均消费中位数": f"{df['人均消费'].median():.0f} 元",
        "评分缺失率": f"{(1-rated_ratio)*100:.1f}%",
        "风险门店占比(C类)": f"{c_count/len(df)*100:.1f}%",
    }

    # ---------- 图表 ----------
    _chart_tier(tier_stats)
    _chart_scatter(rated, comment_median)
    _chart_district_c(df)
    _chart_district_missing(df)
    _chart_cuisine(df)
    _chart_price(df)

    # ---------- 风险清单（脱敏：不含店名/地址/电话） ----------
    risk_cols = ["店铺id", "评分", "评论数量", "人均消费", "归属商圈", "菜系类型", "county", "距离描述"]
    risk = df[df["门店分层"].isin(["C_口碑风险", "C_边缘店"])][risk_cols].copy()
    risk = risk.sort_values(["评分", "评论数量"], ascending=[True, False])
    risk_path = os.path.join(OUT_DIR, "risk_store_list.csv")
    risk.to_csv(risk_path, index=False, encoding="utf-8-sig")

    # ---------- Markdown 报告 ----------
    _write_markdown(df, tier_stats, comment_median, kpi, c_count, risk)

    # ---------- 自包含 HTML 看板 ----------
    html_path = _write_html(df, tier_stats, kpi, risk, comment_median)

    print("\n================ 看板结果 ================")
    for k, v in kpi.items():
        print(f"{k}: {v}")
    print(f"\n风险门店清单（脱敏）: {risk_path}（{len(risk)} 家）")
    print(f"HTML 看板: {html_path}")
    print(f"图表与报告: {OUT_DIR}/")


# ---------------- 图表 ----------------

def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150)
    plt.close(fig)


def _chart_tier(tier_stats):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    colors = {"A_明星店": "#2d6a4f", "B_潜力店": "#74a892", "C_口碑风险": "#e76f51",
              "C_边缘店": "#f4a261", "D_评分缺失": "#b0b3b8"}
    labels = ["A 明星店", "B 潜力店", "C 口碑风险", "C 边缘店", "D 评分缺失"]
    vals = [tier_stats[k] for k in ["A_明星店", "B_潜力店", "C_口碑风险", "C_边缘店", "D_评分缺失"]]
    bars = ax.bar(labels, vals, color=[colors[k] for k in tier_stats.index], width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + len(vals) * 0.008,
                f"{v:,}（{v/vals[0]*100:.1f}% 基准）", ha="center", fontsize=9)
    ax.set_title("门店健康分层分布（评分≥4.5 为高评分，评论量≥中位数为高流量）")
    ax.set_ylabel("门店数")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "01_store_tier.png")


def _chart_scatter(rated, comment_median):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = {"A_明星店": "#2d6a4f", "B_潜力店": "#8ecae6", "C_口碑风险": "#e76f51", "C_边缘店": "#f4a261"}
    for tier, c in colors.items():
        sub = rated[rated["门店分层"] == tier]
        ax.scatter(sub["评论数量"], sub["评分"], s=6, alpha=0.35, c=c, label=tier.split("_", 1)[1], rasterized=True)
    ax.axvline(comment_median, color="gray", ls="--", lw=1)
    ax.axhline(RATING_HIGH, color="gray", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("评论数量（对数轴）")
    ax.set_ylabel("评分")
    ax.set_title("评分 × 评论量四象限（有评分门店）")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "02_score_comment_scatter.png")


def _chart_district_c(df):
    rated = df[df["评分"].notna()]
    grp = rated.groupby("归属商圈")["门店分层"].apply(
        lambda s: (s.isin(["C_口碑风险", "C_边缘店"])).mean()
    ).reset_index()
    grp.columns = ["归属商圈", "c_ratio"]
    cnt = rated["归属商圈"].value_counts()
    grp = grp[grp["归属商圈"].map(cnt) >= 30].sort_values("c_ratio", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.barh(grp["归属商圈"][::-1], grp["c_ratio"][::-1] * 100, color="#e76f51")
    ax.set_xlabel("C 类门店占比（%）")
    ax.set_title("商圈风险 Top15（有评分门店≥30 家，C 类占比）")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, "03_district_c_ratio.png")


def _chart_district_missing(df):
    grp = df.groupby("归属商圈")["评分"].apply(lambda s: s.isna().mean()).reset_index()
    grp.columns = ["归属商圈", "missing_ratio"]
    cnt = df["归属商圈"].value_counts()
    grp = grp[grp["归属商圈"].map(cnt) >= 30].sort_values("missing_ratio", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.barh(grp["归属商圈"][::-1], grp["missing_ratio"][::-1] * 100, color="#b0b3b8")
    ax.set_xlabel("评分缺失率（%）")
    ax.set_title("商圈评分缺失率 Top15（门店≥30 家）")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, "04_district_missing.png")


def _chart_cuisine(df):
    rated = df[df["评分"].notna()]
    grp = rated.groupby("菜系类型").agg(
        平均评分=("评分", "mean"),
        人均消费=("人均消费", "median"),
        门店数=("店铺id", "count"),
    ).reset_index()
    grp = grp[grp["门店数"] >= 200].sort_values("门店数", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    sc = ax.scatter(grp["人均消费"], grp["平均评分"], s=grp["门店数"] / 4, c="#2d6a4f", alpha=0.65)
    for _, r in grp.iterrows():
        ax.annotate(r["菜系类型"], (r["人均消费"], r["平均评分"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("人均消费中位数（元）")
    ax.set_ylabel("平均评分")
    ax.set_title("主要菜系经营画像（门店≥200 家，气泡大小=门店数）")
    ax.grid(alpha=0.3)
    _save(fig, "05_cuisine_profile.png")


def _chart_price(df):
    pos = df[df["人均消费"] > 0]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(pos["人均消费"], bins=60, color="#8ecae6", edgecolor="white")
    ax.axvline(pos["人均消费"].median(), color="#d00000", ls="--", lw=1.4,
               label=f"中位数 {pos['人均消费'].median():.0f} 元")
    ax.set_xlabel("人均消费（元，>0 样本）")
    ax.set_ylabel("门店数")
    ax.set_title(f"人均消费分布（>0 样本 {len(pos):,} 家，0 值占位 {(df['人均消费']==0).mean()*100:.1f}%）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "06_price_distribution.png")


# ---------------- 报告与 HTML ----------------

def _img_b64(name: str) -> str:
    with open(os.path.join(OUT_DIR, name), "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _write_markdown(df, tier_stats, comment_median, kpi, c_count, risk):
    lines = [
        "# 餐饮门店经营诊断看板",
        "",
        "> 基于北京市 4.4 万餐饮商铺真实数据 ｜ 门店健康分层 + 商圈/菜系诊断 + 风险识别",
        "",
        "## 一、分层规则",
        "",
        f"- 高评分：评分 ≥ {RATING_HIGH}；高流量：评论数量 ≥ 中位数 {comment_median:.0f} 条",
        "- A 明星店（高评分 × 高流量）/ B 潜力店（高评分 × 低流量）",
        "- C 口碑风险（低评分 × 高流量）/ C 边缘店（低评分 × 低流量）",
        "- D 评分缺失（无评分门店，多为无评论新店）",
        "",
        "## 二、核心指标",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
    ]
    for k, v in kpi.items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        f"风险门店（C 类）共 **{c_count:,}** 家，占 **{c_count/len(df)*100:.1f}%**。",
        "",
        "## 三、图表",
        "",
        "![门店分层](01_store_tier.png)",
        "",
        "![评分×评论量](02_score_comment_scatter.png)",
        "",
        "![商圈风险](03_district_c_ratio.png)",
        "",
        "![商圈缺失](04_district_missing.png)",
        "",
        "![菜系画像](05_cuisine_profile.png)",
        "",
        "![价格分布](06_price_distribution.png)",
        "",
        "## 四、风险门店清单（脱敏）",
        "",
        f"共 {len(risk):,} 家，完整清单见 `risk_store_list.csv`（仅含分析字段，不含店名/地址/电话）。",
        "",
        "| 店铺id | 评分 | 评论数量 | 人均消费 | 归属商圈 | 菜系类型 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in risk.head(20).iterrows():
        lines.append(f"| {int(r['店铺id'])} | {r['评分']:.1f} | {int(r['评论数量'])} | "
                     f"{int(r['人均消费'])} | {r['归属商圈']} | {r['菜系类型']} |")
    lines += [
        "",
        "## 五、数据与合规说明",
        "",
        "- 看板基于**真实商铺数据**构建，分析结果均为聚合统计，风险清单已脱敏；",
        "- 原始数据不随仓库公开（代码与数据分离），运行方式见 README。",
        "",
    ]
    with open(os.path.join(OUT_DIR, "dashboard_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_html(df, tier_stats, kpi, risk, comment_median) -> str:
    images = "".join(
        f'<img src="data:image/png;base64,{_img_b64(n)}" alt="{n}" style="width:100%;border:1px solid #e5e7eb;border-radius:8px;">'
        for n in ["01_store_tier.png", "02_score_comment_scatter.png",
                  "03_district_c_ratio.png", "04_district_missing.png",
                  "05_cuisine_profile.png", "06_price_distribution.png"]
    )
    kpi_cards = "".join(
        f'<div style="flex:1;min-width:150px;background:#f9fafb;border:1px solid #e5e7eb;'
        f'border-radius:10px;padding:14px;text-align:center;">'
        f'<div style="color:#6b7280;font-size:13px;">{k}</div>'
        f'<div style="font-size:22px;font-weight:700;color:#111827;margin-top:4px;">{v}</div></div>'
        for k, v in kpi.items()
    )
    risk_rows = "".join(
        f"<tr><td>{int(r['店铺id'])}</td><td>{r['评分']:.1f}</td><td>{int(r['评论数量'])}</td>"
        f"<td>{int(r['人均消费'])}</td><td>{r['归属商圈']}</td><td>{r['菜系类型']}</td></tr>"
        for _, r in risk.head(50).iterrows()
    )
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>餐饮门店经营诊断看板</title>
</head>
<body style="font-family:'Microsoft YaHei',sans-serif;margin:0;background:#f3f4f6;color:#111827;">
<div style="background:#111827;color:#fff;padding:22px 28px;">
  <h1 style="margin:0;font-size:22px;">餐饮门店经营诊断看板</h1>
  <div style="margin-top:6px;color:#9ca3af;font-size:13px;">
    北京市 {df.shape[0]:,} 家餐饮商铺 · 分层阈值：高评分≥{RATING_HIGH} / 高评论≥中位数 {comment_median:.0f} 条 · 数据构建时间 {pd.Timestamp.now():%Y-%m-%d}</div>
</div>
<div style="max-width:1180px;margin:0 auto;padding:20px 16px 40px;">
  <div style="display:flex;flex-wrap:wrap;gap:12px;margin-bottom:22px;">{kpi_cards}</div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:18px;">
    {images}
  </div>
  <h2 style="margin-top:30px;font-size:18px;">风险门店清单（脱敏，Top 50）</h2>
  <div style="overflow-x:auto;background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:12px;">
    <table style="border-collapse:collapse;width:100%;font-size:13px;">
      <thead><tr style="background:#f9fafb;">
        <th style="padding:8px;text-align:left;">店铺id</th><th style="padding:8px;">评分</th>
        <th style="padding:8px;">评论数量</th><th style="padding:8px;">人均消费</th>
        <th style="padding:8px;text-align:left;">归属商圈</th><th style="padding:8px;text-align:left;">菜系类型</th>
      </tr></thead>
      <tbody>{risk_rows}</tbody>
    </table>
  </div>
</div>
</body>
</html>"""
    html_path = os.path.join(OUT_DIR, "store_ops_dashboard.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


if __name__ == "__main__":
    main()
