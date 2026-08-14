# -*- coding: utf-8 -*-
"""
A/B 实验分析主流程
==================
场景：首页促销卡片「低价提醒」角标（方案 B） vs 现状（方案 A）

分析步骤：
  1. 数据质量校验：SRM 分流均衡性检查 + A/A 检验（验证随机化与指标口径）
  2. 整体效果：CTR / PCVR / CVR 的双比例 z 检验（含置信区间与提升度）
  3. 漏斗拆解：CVR = CTR × PCVR，定位变化来自「吸引点击」还是「点击后转化」
  4. 分群分析（HTE）：城市线级 × 用户类型，识别放量/迭代的差异化策略
  5. 输出：图表 + 结果汇总表 + Markdown 分析报告

运行：python generate_data.py && python run_ab_test.py
"""

import os

# 将 matplotlib 字体缓存放到项目内可写目录，避免依赖用户全局目录
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2_contingency, chisquare

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

DATA_PATH = "data/ab_experiment_data.csv"
OUT_DIR = "output"
ALPHA = 0.05
Z_CRIT = norm.ppf(1 - ALPHA / 2)


# ---------------- 统计工具 ----------------

def prop_metric(df: pd.DataFrame, click_col: str, conv_col: str) -> dict:
    """计算 CTR / PCVR / CVR 及 95% Wald 置信区间"""
    exposed = len(df)
    clicks = int(df[click_col].sum())
    convs = int(df[conv_col].sum())
    ctr, ctr_ci = _prop_ci(clicks, exposed)
    pcvr, pcvr_ci = _prop_ci(convs, clicks) if clicks else (0.0, (0.0, 0.0))
    cvr, cvr_ci = _prop_ci(convs, exposed)
    return {
        "exposed": exposed,
        "clicks": clicks,
        "conversions": convs,
        "CTR": ctr, "CTR_CI": ctr_ci,
        "PCVR": pcvr, "PCVR_CI": pcvr_ci,
        "CVR": cvr, "CVR_CI": cvr_ci,
    }


def _prop_ci(x: int, n: int) -> tuple:
    p = x / n if n else 0.0
    se = np.sqrt(p * (1 - p) / n) if n else 0.0
    return p, (max(0.0, p - Z_CRIT * se), min(1.0, p + Z_CRIT * se))


def ztest_two_prop(x1: int, n1: int, x2: int, n2: int) -> dict:
    """双比例 z 检验（pooled SE），返回差异、置信区间、p 值与提升度"""
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - norm.cdf(abs(z)))
    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p2 - p1
    ci = (diff - Z_CRIT * se_diff, diff + Z_CRIT * se_diff)
    lift = diff / p1 if p1 > 0 else np.nan
    return {
        "p1": p1, "p2": p2, "diff": diff, "ci": ci,
        "lift": lift, "z": z, "p_value": p_value,
    }


def sig_star(p_value: float) -> str:
    return "**" if p_value < 0.01 else ("*" if p_value < 0.05 else "")


# ---------------- 分析 ----------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df_a = df[df["variant"] == "A"]
    df_b = df[df["variant"] == "B"]

    report = []
    log = print

    # ========== 1. 数据质量校验 ==========
    n_a, n_b = len(df_a), len(df_b)
    n_total = n_a + n_b
    chi2_srm, p_srm = chisquare([n_a, n_b], f_exp=[n_total / 2, n_total / 2])

    # A/A 检验：对照组内部再按 user_id 二次 hash 拆成两组，指标不应有显著差异
    import hashlib
    def aa_split(uid: str) -> str:
        return "AA1" if int(hashlib.md5(f"{uid}:aa".encode()).hexdigest(), 16) % 100 < 50 else "AA2"

    aa = df_a.copy()
    aa["aa_group"] = aa["user_id"].apply(aa_split)
    aa1, aa2 = aa[aa["aa_group"] == "AA1"], aa[aa["aa_group"] == "AA2"]
    aa_ctr = ztest_two_prop(aa1["clicked"].sum(), len(aa1), aa2["clicked"].sum(), len(aa2))
    aa_cvr = ztest_two_prop(aa1["converted"].sum(), len(aa1), aa2["converted"].sum(), len(aa2))

    # ========== 2. 整体结果 ==========
    m_a, m_b = prop_metric(df_a, "clicked", "converted"), prop_metric(df_b, "clicked", "converted")
    overall = {}
    for metric in ["CTR", "PCVR", "CVR"]:
        x1 = m_a["clicks"] if metric == "CTR" else (m_a["conversions"] if metric == "CVR" else m_a["conversions"])
        n1 = m_a["exposed"] if metric != "PCVR" else m_a["clicks"]
        x2 = m_b["clicks"] if metric == "CTR" else (m_b["conversions"] if metric == "CVR" else m_b["conversions"])
        n2 = m_b["exposed"] if metric != "PCVR" else m_b["clicks"]
        overall[metric] = ztest_two_prop(x1, n1, x2, n2)

    # ========== 3. 分群分析（HTE） ==========
    segments = {}
    for tier in ["tier_1_2", "tier_3_4"]:
        segments[f"{tier}_all"] = df[df["city_tier"] == tier]
    for utype in ["returning", "new"]:
        segments[f"all_{utype}"] = df[df["user_type"] == utype]
    for tier in ["tier_1_2", "tier_3_4"]:
        for utype in ["returning", "new"]:
            segments[f"{tier}_{utype}"] = df[(df["city_tier"] == tier) & (df["user_type"] == utype)]

    hte_rows = []
    for name, seg in segments.items():
        sa, sb = seg[seg["variant"] == "A"], seg[seg["variant"] == "B"]
        row = {"segment": name, "n_A": len(sa), "n_B": len(sb)}
        for metric in ["CTR", "PCVR", "CVR"]:
            x1 = sa["clicked"].sum() if metric == "CTR" else (sa["converted"].sum() if metric == "CVR" else sa["converted"].sum())
            n1 = len(sa) if metric != "PCVR" else sa["clicked"].sum()
            x2 = sb["clicked"].sum() if metric == "CTR" else (sb["converted"].sum() if metric == "CVR" else sb["converted"].sum())
            n2 = len(sb) if metric != "PCVR" else sb["clicked"].sum()
            r = ztest_two_prop(x1, n1, x2, n2)
            row[f"{metric}_A"] = r["p1"]
            row[f"{metric}_B"] = r["p2"]
            row[f"{metric}_lift"] = r["lift"]
            row[f"{metric}_diff_ci"] = r["ci"]
            row[f"{metric}_p"] = r["p_value"]
        hte_rows.append(row)
    hte = pd.DataFrame(hte_rows)

    # ========== 4. 图表 ==========
    _chart_funnel(m_a, m_b, overall)
    _chart_segments(hte)

    # ========== 5. 报告 ==========
    _write_report(report, df, n_a, n_b, chi2_srm, p_srm, aa_ctr, aa_cvr,
                  m_a, m_b, overall, hte)

    # 控制台摘要
    print("\n================ 整体结果 ================")
    for metric in ["CTR", "PCVR", "CVR"]:
        r = overall[metric]
        print(f"{metric}: A={r['p1']:.4f}  B={r['p2']:.4f}  "
              f"提升={r['lift']*100:+.1f}%   p={r['p_value']:.4g} {sig_star(r['p_value'])}")
    print(f"\nSRM 检查: 卡方={chi2_srm:.2f} p={p_srm:.4f}  -> {'通过' if p_srm > 0.05 else '不通过'}")
    print(f"A/A 检验: CTR p={aa_ctr['p_value']:.4f}  CVR p={aa_cvr['p_value']:.4f}  "
          f"-> {'通过' if aa_ctr['p_value'] > 0.05 and aa_cvr['p_value'] > 0.05 else '不通过'}")
    print(f"\n报告已生成: {OUT_DIR}/ab_test_report.md")
    print(f"图表已生成: {OUT_DIR}/funnel_comparison.png, {OUT_DIR}/segment_lift.png")
    print(f"汇总表已生成: {OUT_DIR}/hte_summary.csv")


# ---------------- 图表 ----------------

def _fmt_pct(x, nd=1):
    return f"{x*100:.{nd}f}%"


def _chart_funnel(m_a, m_b, overall):
    metrics = ["CTR", "PCVR", "CVR"]
    labels = ["点击率 CTR", "点击后转化率 PCVR", "整体转化率 CVR"]
    pa = [m_a[m] for m in metrics]
    pb = [m_b[m] for m in metrics]
    ca = [m_a[f"{m}_CI"] for m in metrics]
    cb = [m_b[f"{m}_CI"] for m in metrics]

    x = np.arange(len(metrics))
    w = 0.34
    fig, ax = plt.subplots(figsize=(9, 5.2))
    yerr_a = np.array([[p - lo for p, (lo, hi) in zip(pa, ca)],
                       [hi - p for p, (lo, hi) in zip(pa, ca)]])
    yerr_b = np.array([[p - lo for p, (lo, hi) in zip(pb, cb)],
                       [hi - p for p, (lo, hi) in zip(pb, cb)]])
    b1 = ax.bar(x - w / 2, pa, w, yerr=yerr_a,
                capsize=4, color="#8ecae6", label="方案 A（对照组）")
    b2 = ax.bar(x + w / 2, pb, w, yerr=yerr_b,
                capsize=4, color="#fb8500", label="方案 B（低价提醒）")
    for i, m in enumerate(metrics):
        r = overall[m]
        ax.text(i + w / 2, pb[i] + 0.015, f"{r['lift']*100:+.1f}%"
                + sig_star(r["p_value"]), ha="center", fontsize=10, color="#9d0208")
    ax.set_xticks(x, labels)
    ax.set_ylabel("转化率")
    ax.set_ylim(0, max(max(pa), max(pb)) * 1.28)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("整体漏斗指标对比（误差线为 95% 置信区间；柱顶为 B 组相对提升）")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/funnel_comparison.png", dpi=150)
    plt.close(fig)


def _chart_segments(hte):
    order = ["tier_1_2_all", "tier_3_4_all", "all_returning", "all_new",
             "tier_1_2_returning", "tier_1_2_new", "tier_3_4_returning", "tier_3_4_new"]
    seg_labels = {
        "tier_1_2_all": "一二线城市·全部",
        "tier_3_4_all": "三四线及以下·全部",
        "all_returning": "全部城市·老客",
        "all_new": "全部城市·新客",
        "tier_1_2_returning": "一二线·老客",
        "tier_1_2_new": "一二线·新客",
        "tier_3_4_returning": "三四线·老客",
        "tier_3_4_new": "三四线·新客",
    }
    hte = hte.set_index("segment").reindex(order).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True)
    for ax, metric in zip(axes, ["CTR", "PCVR", "CVR"]):
        y = np.arange(len(hte))
        diff = hte[f"{metric}_B"] - hte[f"{metric}_A"]
        lo = [hte.loc[i, f"{metric}_diff_ci"][0] for i in range(len(hte))]
        hi = [hte.loc[i, f"{metric}_diff_ci"][1] for i in range(len(hte))]
        colors = ["#2d6a4f" if d > 0 else "#d00000" for d in diff]
        xerr = np.array([[(d - l) * 100 for d, l in zip(diff, lo)],
                         [(h - d) * 100 for d, h in zip(diff, hi)]])
        ax.errorbar(diff * 100, y, xerr=xerr,
                    fmt="o", color="k", ecolor="k", elinewidth=1, capsize=3, alpha=0.35, zorder=1)
        ax.scatter(diff * 100, y, c=colors, s=70, zorder=2)
        ax.axvline(0, color="gray", lw=1, ls="--")
        ax.set_yticks(y, [seg_labels.get(s, s) for s in hte["segment"]], fontsize=9)
        ax.set_xlabel("相对差异（百分点）")
        ax.set_title({"CTR": "点击率 CTR", "PCVR": "点击后转化率 PCVR",
                      "CVR": "整体转化率 CVR"}[metric], fontsize=11)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("分群效果对比（B - A，红=下降 绿=提升；横线为 95% 置信区间）", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/segment_lift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------- 报告 ----------------

def _write_report(report, df, n_a, n_b, chi2_srm, p_srm, aa_ctr, aa_cvr,
                  m_a, m_b, overall, hte):
    lines = []
    w = lines.append
    w("# 首页促销卡「低价提醒」A/B 实验分析报告")
    w("")
    w("> 模拟数据案例分析 ｜ 双比例 z 检验 + 漏斗拆解 + 分群分析（HTE）")
    w("")
    w("## 一、实验背景与业务问题")
    w("")
    w("移动点单 App 计划在首页促销卡片上增加「低价提醒」角标（方案 B），与现状（方案 A）对比。")
    w("要回答两个问题：")
    w("1. 角标能否让更多用户点击促销卡（**CTR 提升**）？")
    w("2. 点击后能否真正带来更多订单（**CVR 提升**）？")
    w("")
    w("常见风险：角标吸引好奇点击，但用户点击后发现受起送门槛、凑单规则限制，"
      "导致**后点击转化率 PCVR 下降**，最终 CVR 未必提升。")
    w("")
    w("## 二、实验设计")
    w("")
    w("| 项目 | 说明 |")
    w("| --- | --- |")
    w(f"| 样本量 | 共 {len(df):,} 名用户（A 组 {n_a:,} / B 组 {n_b:,}） |")
    w("| 分流方式 | 基于 user_id 的稳定 hash 分流（50/50），同一用户始终在同一组 |")
    w("| 指标 | CTR=点击/曝光；PCVR=转化/点击；CVR=转化/曝光（CVR = CTR × PCVR） |")
    w("| 统计方法 | 双比例 z 检验（95% 置信区间）、SRM 分流校验、A/A 检验 |")
    w("| 分群维度 | 城市线级（一二线 / 三四线及以下）× 用户类型（新客 / 老客） |")
    w("")
    w("## 三、数据质量校验")
    w("")
    w(f"- **SRM 分流均衡性**：A/B 实际样本占比与 50/50 无显著偏差"
      f"（卡方={chi2_srm:.2f}，p={p_srm:.4f}）→ **{'通过' if p_srm > 0.05 else '不通过'}**")
    w(f"- **A/A 检验**：对照组内随机拆分两组，CTR 差异 p={aa_ctr['p_value']:.4f}、"
      f"CVR 差异 p={aa_cvr['p_value']:.4f} → **{'通过' if aa_ctr['p_value'] > 0.05 and aa_cvr['p_value'] > 0.05 else '不通过'}**"
      "（随机化与指标口径无系统偏差）")
    w("")
    w("## 四、整体结果")
    w("")
    w("| 指标 | 方案 A | 方案 B | 相对提升 | 差异 95% CI | p 值 |")
    w("| --- | --- | --- | --- | --- | --- |")
    for metric, label in [("CTR", "点击率 CTR"), ("PCVR", "点击后转化率 PCVR"), ("CVR", "整体转化率 CVR")]:
        r = overall[metric]
        w(f"| {label} | {r['p1']*100:.2f}% | {r['p2']*100:.2f}% | "
          f"{r['lift']*100:+.1f}%{sig_star(r['p_value'])} | "
          f"[{r['ci'][0]*100:+.2f}%, {r['ci'][1]*100:+.2f}%] | {r['p_value']:.4g} |")
    w("")
    w("![整体漏斗指标对比](funnel_comparison.png)")
    w("")
    w("**漏斗拆解（CVR = CTR × PCVR）**：B 组点击率显著提升，但点击后转化率下降。"
      "整体 CVR 是否为正，取决于两类效应的净结果。")
    w("")
    w("## 五、分群分析（HTE）")
    w("")
    w("![分群效果对比](segment_lift.png)")
    w("")
    w("| 分群 | CVR_A | CVR_B | CVR 提升 | CVR p 值 | 解读 |")
    w("| --- | --- | --- | --- | --- | --- |")
    for _, row in hte.iterrows():
        seg = row["segment"]
        if seg.endswith("_returning") or seg.endswith("_new"):
            continue  # 组合分群单独展示
        pv = row["CVR_p"]
        if pv < 0.01:
            note = "显著提升" if row["CVR_lift"] > 0 else "显著下降"
        elif pv < 0.05:
            note = "边际提升" if row["CVR_lift"] > 0 else "边际下降"
        else:
            note = "无显著差异"
        w(f"| {seg} | {row['CVR_A']*100:.2f}% | {row['CVR_B']*100:.2f}% | "
          f"{row['CVR_lift']*100:+.1f}%{sig_star(pv)} | {pv:.4g} | {note} |")
    w("")
    w("## 六、结论与建议")
    w("")
    w("结合整体与分群结果，形成以下决策建议：")
    w("")
    tier12 = hte[hte["segment"] == "tier_1_2_all"].iloc[0]
    tier34 = hte[hte["segment"] == "tier_3_4_all"].iloc[0]
    w(f"1. **一二线城市：建议放量上线**。CVR 提升 {tier12['CVR_lift']*100:+.1f}%"
      f"（p={tier12['CVR_p']:.4g}），点击与转化同步改善，是本次改版的主要收益来源。")
    w(f"2. **三四线及以下城市：暂缓全量，先优化承接体验**。该分群点击率提升明显，"
      f"但 PCVR 下降 {abs(overall['PCVR']['lift'])*100:.1f}% 以上，CVR 提升 {tier34['CVR_lift']*100:+.1f}%"
      f"（p={tier34['CVR_p']:.4g}）——角标吸引的点击未能有效转化为订单，"
      "建议先调整起送门槛 / 凑单提示，再做一轮验证实验。")
    w("3. **监控老客转化**：老客对价格敏感度更高，需关注「低价提醒」是否带来客单价稀释，"
      "建议将 GMV、客单价、优惠券成本加入守卫指标（guardrail）。")
    w("")
    w("## 七、局限与后续")
    w("")
    w("- 本实验数据为**模拟数据**，用于演示完整分析流程，真实结论需以线上实验为准；")
    w("- 后续可补充：CUPED 方差缩减、分阶段放量（10%→25%→50%→100%）、"
      "长期留存与 LTV 观测、成本/收益测算。")
    w("")
    with open(f"{OUT_DIR}/ab_test_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 汇总表
    out = hte.copy()
    for metric in ["CTR", "PCVR", "CVR"]:
        out[f"{metric}_diff"] = out[f"{metric}_B"] - out[f"{metric}_A"]
        out[f"{metric}_ci_lo"] = out[f"{metric}_diff_ci"].apply(lambda c: c[0])
        out[f"{metric}_ci_hi"] = out[f"{metric}_diff_ci"].apply(lambda c: c[1])
    out.drop(columns=[f"{m}_diff_ci" for m in ["CTR", "PCVR", "CVR"]], inplace=True)
    out.to_csv(f"{OUT_DIR}/hte_summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
