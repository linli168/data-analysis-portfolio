# -*- coding: utf-8 -*-
"""
共享基础模块
  1. 报告输出：init_report / close_report / P（同时写入控制台与 报告.txt）
  2. 全局常量：DISTRICT_MAP 区域分组映射
  3. 通用文本工具：q2b 全角转半角、parse_scores 解析多维评分、calc_biz_hours 营业时长解析
  4. 统计工具：pool_r 相关系数 Rubin 合并（Fisher z）
  5. 绘图中文字体与全局画布配置
"""
import pandas as pd, numpy as np, ast, re, sys, os, warnings
warnings.filterwarnings('ignore')

# ---- 统一输出编码：Windows控制台默认GBK无法编码部分特殊字符(如"伪R²"上标) ----
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ---- 绘图全局配置 ----
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'DengXian']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.weight'] = 'normal'
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 5)


# ===================== 报告输出 =====================
_output = None


def init_report(path='报告.txt'):
    """初始化报告文件（覆盖写入）"""
    global _output
    _output = open(path, 'w', encoding='utf-8')


def close_report():
    """关闭报告文件"""
    global _output
    if _output is not None:
        _output.close()
        _output = None


def P(*a, **kw):
    """打印到控制台 + 报告文件（与旧版单脚本行为一致）"""
    print(*a, **kw, file=sys.stdout, flush=True)
    if _output is not None:
        print(*a, **kw, file=_output, flush=True)


# ===================== 全局常量 =====================
DISTRICT_MAP = {'东城区': '核心城区', '西城区': '核心城区', '朝阳区': '核心城区', '海淀区': '核心城区',
                '丰台区': '核心城区', '石景山区': '核心城区', '通州区': '近郊区', '大兴区': '近郊区',
                '昌平区': '近郊区', '顺义区': '近郊区', '房山区': '近郊区', '门头沟区': '远郊区',
                '怀柔区': '远郊区', '平谷区': '远郊区', '密云区': '远郊区', '延庆区': '远郊区'}


# ===================== 通用文本工具 =====================
def parse_scores(s):
    """解析 scores JSON 字符串为 {维度名: 分数} 字典"""
    try:
        items = ast.literal_eval(str(s))
        return {i['name']: i['score'] for i in items if isinstance(i, dict) and 'name' in i and 'score' in i}
    except Exception:
        return {}


def q2b(s):
    """全角字符转半角"""
    res = []
    for uchar in s:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif 65281 <= inside_code <= 65374:
            inside_code -= 65248
        res.append(chr(inside_code))
    return ''.join(res)


def calc_biz_hours(t):
    """从营业时间文本解析出标准化营业时长(小时)；无法解析返回 NaN"""
    if pd.isna(t) or str(t).strip() in ['未知', '', 'nan', '休息中']:
        return np.nan
    t = str(t).strip()
    t = q2b(t).lower()
    t = re.sub(r'[\^#?]+', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    t = re.sub(r'^(营业中|休息中|明天)\s*', '', t)
    t = re.sub(r'\s*营业$', '', t).strip()
    if not t:
        return np.nan
    if re.search(r'全天|24\s*小时|24\s*h', t):
        return 24.0
    total = 0.0
    valid = 0
    pattern = r'(\d{1,2})[:\.](\d{1,2})\s*[-~至到]\s*(\d{1,2})[:\.](\d{1,2})'
    for m in re.findall(pattern, t):
        h1, m1 = min(int(m[0]), 23), min(int(m[1]), 59)
        h2, m2 = min(int(m[2]), 23), min(int(m[3]), 59)
        start = h1 * 60 + m1
        end = h2 * 60 + m2
        if end <= start:
            end += 1440
        total += (end - start) / 60
        valid += 1
    return round(total, 1) if valid > 0 else np.nan


# ===================== 统计工具 =====================
def pool_r(rs):
    """Rubin规则合并相关系数：Fisher z变换取均值再反变换"""
    z = [np.arctanh(np.clip(float(r), -0.999, 0.999)) for r in rs]
    return float(np.tanh(np.mean(z))), float(np.min(rs)), float(np.max(rs))
