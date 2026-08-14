# -*- coding: utf-8 -*-
"""
模块：missing_diagnosis —— 缺失机制分析
  对应报告【二】缺失机制分析（含MNAR影响论证）
  三层递进：分组对比 → t/卡方检验 → Logistic回归（McFadden伪R²）
  返回承载后续全流程的 sd（含 人均消费 float 与 区域分组）
"""
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from .common import P, DISTRICT_MAP


def diagnose_missing(shop):
    """对评分/商圈/菜系/营业时间四类字段执行缺失机制统计检验
    返回 sd = shop副本 + 人均消费(float) + 区域分组
    """
    sd = shop.copy()
    sd['人均消费'] = sd['人均消费'].astype(float)
    sd['区域分组'] = sd['county'].map(DISTRICT_MAP).fillna('其他')

    has_s, no_s = sd['评分'].notna(), sd['评分'].isnull()
    P(f'  有评分: {has_s.sum()}家 - 人均{sd.loc[has_s, "人均消费"].mean():.1f}元, 评论{sd.loc[has_s, "评论数量"].mean():.1f}条')
    P(f'  无评分: {no_s.sum()}家 - 人均{sd.loc[no_s, "人均消费"].mean():.1f}元, 评论{sd.loc[no_s, "评论数量"].mean():.1f}条')

    # 图2：有无评分对比柱状图
    avg_cost = [sd.loc[has_s, "人均消费"].mean(), sd.loc[no_s, "人均消费"].mean()]
    avg_comment_c = [sd.loc[has_s, "评论数量"].mean(), sd.loc[no_s, "评论数量"].mean()]
    group_name = ['有评分', '无评分']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(2)
    width = 0.35
    ax1.bar(x, avg_cost, width, label=group_name, color=['#4a88cf', '#e66b6b'])
    ax1.set_title('有/无评分店铺人均消费对比')
    ax1.set_ylabel('人均消费(元)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(group_name)
    ax1.grid(axis='y', alpha=0.2)
    ax2.bar(x, avg_comment_c, width, label=group_name, color=['#4a88cf', '#e66b6b'])
    ax2.set_title('有/无评分店铺平均评论数对比')
    ax2.set_ylabel('评论条数')
    ax2.set_xticks(x)
    ax2.set_xticklabels(group_name)
    ax2.grid(axis='y', alpha=0.2)
    fig.suptitle('图2 有无评分店铺指标对比', y=0.98)
    plt.tight_layout()
    plt.savefig('图2_有无评分对比柱状图.png')
    plt.close()
    P('已生成【图2】有无评分店铺多指标对比图')

    # 图3：六边形密度图
    plt.figure(figsize=(9, 5))
    sns.jointplot(data=sd[sd['评分'].notna()], x='评论数量', y='评分', kind='hex', height=5)
    plt.suptitle('图3 评论数量与评分六边形密度分布', y=1.02)
    plt.savefig('图3_评论数vs评分六边形密度图.png', bbox_inches='tight')
    plt.close()
    P('已生成【图3】评论数vs评分密度图')

    # ---- 统计检验 ----
    P('\n--- 缺失机制统计检验 ---')
    sd_temp = sd.copy()
    sd_temp['人均消费_clip'] = sd_temp['人均消费'].replace(0, np.nan)
    sd_temp['人均消费_clip'] = sd_temp['人均消费_clip'].clip(lower=1, upper=500)
    sd_temp['评论数_log'] = np.log1p(sd_temp['评论数量'])
    target_cols = ['评分', '归属商圈', '菜系类型', '营业时间']
    numeric_vars = ['人均消费_clip', '评论数_log', '经度', '纬度']
    categorical_var = '区域分组'

    for col in target_cols:
        P(f'\n--- {col}缺失机制检验 ---')
        mask = sd_temp[col].isnull()
        P(f'  缺失样本量：{mask.sum()} / {len(mask)} ({mask.sum()/len(mask)*100:.1f}%)')
        for num in numeric_vars:
            g1 = sd_temp.loc[mask, num].dropna()
            g2 = sd_temp.loc[~mask, num].dropna()
            if len(g1) > 0 and len(g2) > 0:
                stat, p = ttest_ind(g1, g2, equal_var=False)
                P(f'    {num:12}: t={stat:7.3f}, p={p:.5f} -> {"显著" if p < 0.05 else "不显著"}')
        cross = pd.crosstab(mask, sd_temp[categorical_var])
        if cross.shape[1] > 1:
            chi2, p, dof, ex = chi2_contingency(cross)
            P(f'  区域分组（卡方）：chi2={chi2:.3f}, p={p:.5f} -> {"显著" if p < 0.05 else "不显著"}')
        X_numeric = sd_temp[['人均消费_clip', '评论数_log', '经度', '纬度']].copy()
        X_dummies = pd.get_dummies(sd_temp[categorical_var], prefix='region', drop_first=True)
        X = pd.concat([X_numeric, X_dummies], axis=1)
        y = mask.astype(int)
        reg_data = pd.concat([X, y.rename('y')], axis=1).dropna()
        X_clean = reg_data.drop(columns='y')
        y_clean = reg_data['y']
        X_scaled = StandardScaler().fit_transform(X_clean)
        X_scaled = pd.DataFrame(X_scaled, columns=X_clean.columns, index=X_clean.index)
        X_scaled = sm.add_constant(X_scaled)
        model = sm.Logit(y_clean, X_scaled).fit(method='bfgs', maxiter=1000, disp=False)
        P(f'  Logistic有效样本量：{len(reg_data)}，McFadden伪R²={model.prsquared:.4f}')
        coef_comment = model.params['评论数_log']
        p_comment = model.pvalues['评论数_log']
        P(f'  评论数_log系数={coef_comment:.4f}, p={p_comment:.5f}')
    P('\n--- 缺失机制检验完成 ---')

    return sd
