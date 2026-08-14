# -*- coding: utf-8 -*-
"""
模块：data_loader —— 数据加载、去重、时间完整性、基线统计
  对应报告【一】数据基线 + 商铺去重 + 时间完整性
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .common import P

SHOP_PATH = '北京市餐饮商铺数据(4.4w).xlsx'
COMMENT_PATH = '北京市餐饮商铺评论数据(21w).xlsx'


def load_data(shop_path=SHOP_PATH, comment_path=COMMENT_PATH):
    """读取两个 Excel 数据表并打印维度"""
    shop = pd.read_excel(shop_path)
    comment = pd.read_excel(comment_path)
    P(f'\n商铺表: {shop.shape[0]}行 x {shop.shape[1]}列  |  评论表: {comment.shape[0]}行 x {comment.shape[1]}列')
    return shop, comment


def clean_basics(shop, comment):
    """商铺去重 + 评论时间完整性校验 + 处理前基线统计 + 图1缺失率条形图
    返回 (去重后shop, comment, 精确重复条数)
    """
    # ---- 商铺去重 ----
    P('\n--- 商铺去重 ---')
    shop_dup_exact = shop.duplicated(subset=['店铺名称', '地址'], keep='first')
    P(f'精确重复(同名同址): {shop_dup_exact.sum()}条')
    shop = shop[~shop_dup_exact].reset_index(drop=True)
    P(f'去重后商铺: {len(shop)}家')
    shop_dup_exact_sum = int(shop_dup_exact.sum())

    # ---- 评论时间完整性校验 ----
    P('\n--- 评论时间完整性校验 ---')
    comment['评论时间_parsed'] = comment['pag_time'].apply(
        lambda x: pd.to_datetime(str(x).replace('发布点评', '').strip(), errors='coerce'))
    now_ts = pd.Timestamp('2024-06-30')
    P(f'未来时间评论: {(comment["评论时间_parsed"] > now_ts).sum()}条')
    P(f'10年前评论: {(comment["评论时间_parsed"] < pd.Timestamp("2010-01-01")).sum()}条')

    # ---- 处理前基线 ----
    P('\n--- 处理前基线 ---')
    null_info = dict()
    for col in shop.columns:
        nulls = shop[col].isnull().sum()
        miss_pct = nulls / len(shop) * 100
        null_info[col] = miss_pct
        dt = shop[col].dtype
        if dt in ['int64', 'float64']:
            P(f'  {col}: 缺失{nulls}({miss_pct:.1f}%), 均值{shop[col].dropna().mean():.2f}, std{shop[col].dropna().std():.3f}')
        else:
            P(f'  {col}: 缺失{nulls}({miss_pct:.1f}%), 唯一{shop[col].nunique()}')

    # 图1：缺失率条形图
    plt.figure(figsize=(10, 4.5))
    miss_cols = ['米其林推荐指数', '评分', '营业时间', '归属商圈', '店铺电话', '菜系类型']
    miss_rate = [null_info[c] for c in miss_cols]
    sns.barplot(x=miss_cols, y=miss_rate, palette='Reds')
    plt.title('图1 处理前各字段缺失率分布', fontsize=12)
    plt.ylabel('缺失率(%)')
    plt.xticks(rotation=30)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('图1_字段缺失率条形图.png')
    plt.close()
    P('已生成【图1】处理前字段缺失率条形图')

    # 辅助图：评论季度分布
    comment['季度'] = comment['评论时间_parsed'].dt.quarter
    quarter_cnt = comment['季度'].value_counts().sort_index()
    plt.figure(figsize=(8, 4))
    sns.barplot(x=quarter_cnt.index, y=quarter_cnt.values)
    plt.title('辅助图 评论季度分布')
    plt.xlabel('季度')
    plt.ylabel('评论数量')
    plt.savefig('辅助_评论季度分布.png')
    plt.close()

    return shop, comment, shop_dup_exact_sum
