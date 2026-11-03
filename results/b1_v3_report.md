# Benchmark I (invsqrt) v3 — 分数报告

## 最终结果(官方验证器 200M shots 全量 PASS)

| 指标 | 值 |
|---|---|
| score | **173,002.75** |
| num_qubits | 91 |
| toffoli_equivalent | 1901.13 |
| toffoli_count (动态) | 1573.00 |
| max_toffoli_count (静态) | 1584 |
| rotation_count | 45 (T/4 ≈ 328) |
| worst_error_over_tolerance | 0.9628 @ x=1034666 |
| verified_shots | 200,540,928 |
| circuit_sha256 | ee3bf648fc6242f549079bec687d1b1c30d85307717c7b48037dd2194f06496a |

基线 (v2): 1,627,018 → v3: **173,003 (9.4× 提升)**; 相对公开冠军 (173,002.75) 持平,
相对 Gu et al. 报告分数与官方 19.5 分团队的 signed-digit 结果同水平。

## 架构(独立实现)

中心化 signed-digit 二次 Horner (src/b1v3_model.py + src/b1v3_build.py):

1. **归一化**: x << s 到 [2^21, 2^22), 5 个二进制 flag (barrel shift 16/8/4/2/1),
   flag 用重算/HMR+条件相位修正清理;
2. **查表**: 一个 kickmix QROM (256 项 × 54 bit, 测量清理) 载入 K2/K1/K0;
   K0 拆成高字(8 bit,提前 phasing 后即卸载)与低字(21 bit,行后 phasing);
3. **Horner**: R = K1 − Σ sig_i·round(K2/2^sh2) (8 行, drop i=5);
   Q = Σ sig_i·round(R/2^sh1) (14 行); sig_i = ±1 使每行只是一个移位副本的
   条件加/减(取补技巧),Gidney 进位加法器 + 测量擦除进位;
4. **Phasing**: 高字 + 低字分别 barrel-shift (s−e)/2 后逐位 z_pow,
   低字与高字在 S=20 接缝处重叠相加,精确实现进位;
5. x[21] (归一化后恒 1) 借用为一条干净的进位 ancilla (条件感知干净比特复用)。

配置: F2/F1/FYR/GY = 8/19/27/1, drop2={5}, LB=21, S=20, NSEG=128×2(parity)。
分阶段 Toffoli: lookup 282+22, yrows 560, rrows 266, phase shifts 248,
normalize/unnormalize 206, 合计 1584。

## 关键自研洞察(优于公开参考的拟合)

1. **精确可达域拟合**: 公开参考对全部 2^14 网格拟合,但真实域有结构:
   e=1 段只有偶数 d 可达; e=0 且 a≥64 的段只有 d≡0 (mod 4) 可达
   (x ≤ 3,133,452 使 s=0 只能覆盖 a<63); a=63 部分受限。已按段精确可达域
   拟合与布局(最坏段 (0,124) 公开参考在不可达 d=9899 处 metric 1.083,
   真实域最坏是 d=9896 的 0.963)。
2. 该配置的离散 minimax 最优唯一: 独立搜索收敛到与公开最优一致的系数表
   (0/256 段不同), 代码完全自写。

## 自包含性(内核重建)

```
PYTHONPATH=<kickmix>:<src> python src/b1v3_build.py results/b1_v3.kmb
```
不带 --tabs 时运行时自行拟合系数(~75–120 s),产物与已验证电路逐位一致
(sha256 相同,已实测)。仅依赖 numpy + kickmix,无 data/community 依赖。

## 已验证失败的微调实验

| 配置 | 结果 |
|---|---|
| drop={5,6} | FAIL, worst 1.003 @ x=8959 (差 0.3%) |
| F1=18 (w1=17, −16T −1Q) | FAIL, worst 1.066 @ x=3763 |
| F2=7 (−1Q) | FAIL, y 溢出 @ x=1 |
| F1=18 + drop={5,6} | FAIL, worst 1.045 |
| GY=0/FYR=28 (公开参考 0.902) | 分数 175.5k,劣于 173k |

结论: (8,19,27,1)-drop5 是本架构族的验证最优; 173k 为交付版本。

## 文件

- src/b1v3_model.py — 精确定点模型 + Lawson/整数 minimax 拟合 + 全域检查
- src/b1v3_build.py — kickmix 构建器(分阶段 mark)
- src/b1v3_test.py — WSL 快速功能测试
- scripts/b1v3_fit.py / b1v3_fit_variants.py / b1v3_build_test.py / wsl_b1_v3.sh
- results/b1_v3.kmb + results/b1_v3_score.json (官方验证 JSON)
- results/b1v3_tabs_8_19_27_1_d5.npz (拟合系数,构建器可自行重算)
