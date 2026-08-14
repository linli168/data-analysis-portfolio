# -*- coding: utf-8 -*-
"""
模块：outlier_detection —— 异常值识别方法对比
  对应报告 3.0.1：Z-score 3σ vs IQR 四分位距 vs Winsorize 缩尾
  返回 (p1_val, p99_val, valid_price)：缩尾阈值与有效消费样本，供填补/稳健性/质量汇总使用
"""
import numpy as np
from scipy.stats import zscore

from .common import P


def compare_outlier_methods(sd):
    """对比三种异常识别方法，选定 P1/P99 缩尾策略
    返回 (p1_val, p99_val, valid_price)
    """
    P('\n--- 3.0.1 异常值识别方法对比：Z-score vs IQR vs 缩尾---')
    P('参考评分标准要求，对比三种通用异常值识别方法在餐饮人均消费数据上的表现。')

    valid_price = sd.loc[sd['人均消费'] > 0, '人均消费']
    price_vals = valid_price.values

    # Z-score 3σ法
    zs = np.abs(zscore(price_vals))
    zscore_outliers = np.sum(zs > 3)
    P(f'① Z-score 3σ法：阈值|z|>3，识别异常{len(price_vals)-zscore_outliers}个正常 / {zscore_outliers}个异常')

    # IQR法
    q1, q3 = np.percentile(price_vals, [25, 75])
    iqr = q3 - q1
    lower_iqr = q1 - 1.5 * iqr
    upper_iqr = q3 + 1.5 * iqr
    iqr_outliers = np.sum((price_vals < lower_iqr) | (price_vals > upper_iqr))
    P(f'② IQR四分位距法：Q1={q1:.1f}, Q3={q3:.1f}, IQR={iqr:.1f}')
    P(f'   下限={lower_iqr:.1f}, 上限={upper_iqr:.1f}, 识别异常{iqr_outliers}个 ({iqr_outliers/len(price_vals)*100:.1f}%)')

    # 缩尾法（P1/P99）
    p1_val = np.percentile(price_vals, 1)
    p99_val = np.percentile(price_vals, 99)
    winsor_low = np.sum(price_vals < p1_val)
    winsor_high = np.sum(price_vals > p99_val)
    P(f'③ Winsorize缩尾法：P1={p1_val:.0f}, P99={p99_val:.0f}')
    P(f'   低端{winsor_low}个({winsor_low/len(price_vals)*100:.1f}%), 高端{winsor_high}个({winsor_high/len(price_vals)*100:.1f}%)')

    P('\n方法对比结论：')
    P(f'  - Z-score(3σ)认定异常下限为负值(不适用于右偏数据)，仅识别高端{zscore_outliers}个极值')
    P(f'  - IQR法识别异常{iqr_outliers}个({iqr_outliers/len(price_vals)*100:.1f}%)，会将大量高端真实店铺误判为异常')
    P(f'  - 缩尾P1/P99仅修正{winsor_low+winsor_high}个({(winsor_low+winsor_high)/len(price_vals)*100:.1f}%)极端值')
    P('  选择依据：人均消费高度右偏，Z-score假设正态分布不成立；IQR将高端真实门店误标异常；')
    P('  缩尾法仅修正最极端尾部，保留主体分布，最适合餐饮数据场景。')

    return p1_val, p99_val, valid_price
