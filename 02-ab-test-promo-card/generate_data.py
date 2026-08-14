# -*- coding: utf-8 -*-
"""
生成 A/B 实验模拟数据
======================
业务场景：外卖/咖啡点单 App 首页促销卡片改版
- 方案 A（对照组）：现有促销卡片
- 方案 B（实验组）：促销卡片增加「低价提醒」角标

模拟机制（符合真实业务的假设）：
- B 组角标会吸引更多用户点击 -> CTR 提升
- 但部分用户点击后发现受「起送门槛 / 凑单规则」限制 -> 后点击转化率 PCVR 下降
- 不同城市线级 / 用户类型受影响程度不同

输出：
- data/ab_experiment_data.csv   完整实验数据（约 20 万用户，可复现）
- sample_ab_data.csv            前 200 行示例数据（入库展示结构）
"""

import hashlib
import os

import numpy as np
import pandas as pd

N_USERS = 200_000
SEED = 42
OUT_DIR = "data"
SAMPLE_ROWS = 200


def stable_split(user_id: str, bucket: int, pct: int = 50) -> bool:
    """基于 user_id 的稳定 hash 分流：同一用户永远进入同一组（模拟真实实验平台）"""
    h = int(hashlib.md5(f"{user_id}:{bucket}".encode("utf-8")).hexdigest(), 16)
    return h % 100 < pct


def main():
    rng = np.random.default_rng(SEED)

    # ---------- 用户画像 ----------
    n1 = int(N_USERS * 0.60)  # 一二线城市
    n2 = N_USERS - n1         # 三四线及以下
    city_tier = np.array(["tier_1_2"] * n1 + ["tier_3_4"] * n2)
    rng.shuffle(city_tier)

    user_type = np.where(
        rng.random(N_USERS) < 0.55, "returning", "new"
    )
    user_id = np.array([f"U{1000000 + i}" for i in range(N_USERS)])

    df = pd.DataFrame({"user_id": user_id, "city_tier": city_tier, "user_type": user_type})

    # ---------- 分流：A / B ----------
    df["variant"] = np.where(
        [stable_split(uid, bucket=1) for uid in df["user_id"]], "B", "A"
    )

    # ---------- 行为概率（真实业务假设的数值化） ----------
    # 点击率 CTR：B 角标带来提升；老用户点击率略高
    ctr_base = df["city_tier"].map({"tier_1_2": 0.135, "tier_3_4": 0.105})
    ctr_user = df["user_type"].map({"returning": 1.05, "new": 0.95})
    ctr_mult = np.where(df["variant"] == "B",
                        df["city_tier"].map({"tier_1_2": 1.28, "tier_3_4": 1.32}),
                        1.0)
    p_click = (ctr_base * ctr_user * ctr_mult).to_numpy()

    # 后点击转化率 PCVR：B 组在一二线基本持平，在三四线因起送门槛明显下降
    pcvr_base = df["city_tier"].map({"tier_1_2": 0.330, "tier_3_4": 0.290})
    pcvr_user = df["user_type"].map({"returning": 1.10, "new": 0.90})
    pcvr_mult = np.where(df["variant"] == "B",
                         df["city_tier"].map({"tier_1_2": 0.97, "tier_3_4": 0.72}),
                         1.0)
    p_convert_given_click = (pcvr_base * pcvr_user * pcvr_mult).to_numpy()

    # ---------- 行为模拟 ----------
    df["clicked"] = rng.binomial(1, p_click).astype(int)
    df["converted"] = np.where(
        df["clicked"] == 1,
        rng.binomial(1, p_convert_given_click).astype(int),
        0,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "ab_experiment_data.csv"),
              index=False, encoding="utf-8")
    df.head(SAMPLE_ROWS).to_csv("sample_ab_data.csv", index=False, encoding="utf-8")

    print(f"已生成完整实验数据：{len(df)} 行 -> {OUT_DIR}/ab_experiment_data.csv")
    print(f"示例数据（前 {SAMPLE_ROWS} 行）-> sample_ab_data.csv")
    print(df[["variant", "city_tier", "user_type"]].value_counts(normalize=True).round(4))


if __name__ == "__main__":
    main()
