# -*- coding: utf-8 -*-
"""餐饮数据预处理项目 —— 模块化代码包

入口：项目根目录运行  py main.py
模块：
  common              共享基础（报告输出/常量/文本工具/统计合并）
  data_loader         数据加载、去重、基线统计
  missing_diagnosis   缺失机制分析
  outlier_detection   异常值识别方法对比
  imputation          人均消费/营业时长填补、MCAR模拟、CV/KNN
  feature_engineering 编码、文本清洗、复合特征、特征筛选/PCA、一致性校验
  multiple_imputation MICE多重插补 + 主分析
  sentiment_analysis  情感分析 + 偏相关
"""
