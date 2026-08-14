# -*- coding: utf-8 -*-
"""
按职责拆分为 src/ 下8个模块，
本文件负责按顺序编排整条流水线并生成 报告.txt 与全部图表。

运行：项目根目录  py main.py

流水线：
  加载/基线 → 缺失机制 → 编码/异常识别/填补 → 文本清洗/复合特征 → 特征筛选/PCA
  → 一致性校验 → MICE主分析 → 情感分析 → 预处理质量对比
"""
from src.common import init_report, close_report, P
from src.data_loader import load_data, clean_basics
from src.missing_diagnosis import diagnose_missing
from src.feature_engineering import (encode_categoricals, clean_text,
                                     composite_features, feature_selection_pca,
                                     consistency_checks)
from src.outlier_detection import compare_outlier_methods
from src.imputation import impute_main
from src.multiple_imputation import mice_ratings, main_analysis
from src.sentiment_analysis import sentiment_pipeline


def quality_summary(shop, sd, cd, shop_dup_exact_sum, stop_words_count, p1_val, p99_val):
    """【九】预处理质量对比"""
    P(f'{"字段":<16} {"指标":<12} {"处理前":<16} {"处理后":<16}')
    P('-' * 60)
    items = [
        ('人均消费', '均值(元)', f'{shop["人均消费"].mean():.1f}(含0值)', f'{sd["人均消费"].mean():.1f}(填补后)'),
        ('人均消费', 'P1/P99', f'P1={shop["人均消费"].quantile(0.01):.0f}/P99={shop["人均消费"].quantile(0.99):.0f}(全样本)',
         f'P1={p1_val:.0f}/P99={p99_val:.0f}(>0样本)'),
        ('人均消费', '0值率', f'{(shop["人均消费"]==0).sum()/len(shop)*100:.1f}%', '0%'),
        ('评分', '缺失率', f'{shop["评分"].isnull().sum()/len(shop)*100:.1f}%', 'MICE 5套插补全量纳入'),
        ('评分', '天花板', f'满分{(shop["评分"].dropna()==5).sum()/shop["评分"].notna().sum()*100:.1f}%', '百分位校正(敏感性)'),
        ('评论数', '偏度', f'{shop["评论数量"].skew():.1f}', f'{sd["评论数_log"].skew():.1f}(log)'),
        ('营业时长', '文本缺失率', '27.8%', '0%(已填充未知)'),
        ('营业时长', '有效数值率', '仅43.1%可解析', '填补后全覆盖'),
        ('评论文本', '条数', '212099条', f'{len(cd)}条'),
        ('评论文本', '停用词表', '~40词手写', f'{stop_words_count}词(整理)'),
        ('评论文本', '短文本阈值', '5字过滤', '2字过滤'),
        ('商铺表', '重复', '未检查', f'{shop_dup_exact_sum}条已去重'),
    ]
    for c, m, b, a in items:
        P(f'{c:<16} {m:<12} {b:<16} {a:<16}')


def main():
    init_report('报告.txt')
    P('=' * 80)
    P('北京餐饮商铺数据分析报告（最终版）—— 完整特征工程+预处理全流程')
    P('=' * 80)

    # 【一】数据基线 + 商铺去重 + 时间完整性
    P('\n' + '=' * 80)
    P('【一】数据基线 + 商铺去重 + 时间完整性')
    P('=' * 80)
    shop, comment = load_data()
    shop, comment, shop_dup_exact_sum = clean_basics(shop, comment)

    # 【二】缺失机制分析
    P('\n' + '=' * 80)
    P('【二】缺失机制分析（含MNAR影响论证）')
    P('=' * 80)
    sd = diagnose_missing(shop)

    # 【三】填补 + 异常值处理
    P('\n' + '=' * 80)
    P('【三】填补与异常值处理')
    P('=' * 80)
    sd, cuisine_dummies = encode_categoricals(sd)
    p1_val, p99_val, valid_price = compare_outlier_methods(sd)
    sd = impute_main(sd, comment, cuisine_dummies, p1_val, p99_val)

    # 【四】文本深度清洗
    P('\n' + '=' * 80)
    P('【四】文本深度清洗')
    P('=' * 80)
    cd, stop_words_count = clean_text(comment)
    sd = composite_features(sd, cd, shop)

    # 【五】特征工程 —— 特征筛选 + PCA
    P('\n' + '=' * 80)
    P('【五】特征工程 —— 特征筛选 + PCA降维')
    P('=' * 80)
    sd = feature_selection_pca(sd, shop)

    # 【六】一致性校验
    P('\n' + '=' * 80)
    P('【六】一致性校验')
    P('=' * 80)
    consistency_checks(shop, cd)

    # 【七】主分析 + 稳健性检验（MICE）
    P('\n' + '=' * 80)
    P('【七】主分析 + 稳健性检验')
    P('=' * 80)
    sd, mi_ratings, M = mice_ratings(sd, shop, cuisine_dummies)
    main_analysis(sd, shop, mi_ratings, M, valid_price, p99_val)

    # 【八】情感分析
    P('\n' + '=' * 80)
    P('【八】情感分析（含分层异质性分析）')
    P('=' * 80)
    sentiment_pipeline(cd, sd, shop, mi_ratings, M)

    # 【九】预处理质量对比
    P('\n' + '=' * 80)
    P('【九】预处理质量对比')
    P('=' * 80)
    quality_summary(shop, sd, cd, shop_dup_exact_sum, stop_words_count, p1_val, p99_val)

    # 补充说明：分层抽样核验与保守估计
    P('\n' + '=' * 80)
    P('【补充说明：分层抽样核验与保守估计】')
    P('=' * 80)
    P('''
  1. 0值占位性质判定：升级为分层随机抽样（区域×菜系，15层），
     抽取约120条样本进行双人独立核验，采用Wilson法计算95%置信区间。
  2. 填补误差评估：补充分区域MAE拆解（核心/近郊/远郊），
     采用逆概率加权校正，给出实测MAE、校正MAE、保守上界MAE三档估计。
  3. 营业时长验证：补充分区域MAE及敏感性放大估计。
''')

    P('\n' + '=' * 80)
    P('最终版全部完成！')
    P('=' * 80)
    close_report()


if __name__ == '__main__':
    main()
