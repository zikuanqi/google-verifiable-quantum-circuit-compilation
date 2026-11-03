# Google Quantum AI — Verifiable Quantum Circuit Compilation (Kaggle)

比赛页面: https://www.kaggle.com/competitions/google-verifiable-quantum-circuit-compilation
规则页(需手动接受): https://www.kaggle.com/competitions/google-verifiable-quantum-circuit-compilation/rules
论文: [arXiv:2610.23832](https://arxiv.org/abs/2610.23832) "Leveraging verifiable structure in quantum circuits to find more efficient compilations with agentic AI" (本地: `docs/verifiable_circuit_compilation_arxiv.pdf`, 全文 `docs/paper_fulltext.txt`)

## 比赛概览

Google Quantum AI 举办的比赛,总奖金 $150,000。目标:优化**容错量子算法中占主导成本的子程序电路**,降低**空间-时间成本(qubits × Toffoli)**。核心工具是 **kickmix** —— 一个 C++20/Python 的电路构建与经典模拟库(Qualtran 的一部分)。

### kickmix 电路模型(为什么可以经典验证)

kickmix 电路只包含四类指令,任何计算基输入只会映射到单一计算基状态(至多带一个相位),因此可以被经典模拟器**线性时间**仿真:

1. **可逆经典门**: X, CX, CCX, SWAP
2. **对角相位门**: Z, CZ, CCZ, Z_POW, NEG
3. **测量/重置**: HMR(哈德玛+测量+重置,产生随机相位 kickback)、R(重置)
4. **经典位操作**: BIT_STORE0/1, BIT_INVERT

所有指令支持条件控制(PUSH_CONDITION / POP_CONDITION 与 per-gate `if b`)。经典模拟器按位打包,SIMD 并行 64–512 个 trajectory,单核可达 6×10^10 指令/秒。

### 验证方式(Fiat-Shamir 模糊测试)

- 提交文件是一个 `.kmb` 二进制电路文件。
- 验证器对电路文件做 **SHA-256**,以摘要(+私有盐)播种 PRNG 生成测试输入 —— **无法针对固定测试集过拟合**。
- 每个输入模拟 `shots_per_x` 次(随机化 HMR/R 的测量结果),检查:
  1. 输入/输出寄存器保持正确经典比特串;
  2. 所有临时 qubit 干净归零;
  3. 累计输出相位在容差内。
- 失败率 ε 经 Clopper-Pearson 上界 ε⁺(置信 1−2⁻¹²⁸)约束,用于部分评分或拒绝提交。

## 三个 Benchmark

### Benchmark I — invsqrt(1/√x 相位,核聚变 Coulomb 模拟)
- 合同: `|x⟩|0⟩ → exp(2πiβ/√x)|x⟩|0⟩`,所有 x ∈ [1, 3,133,452],np = 10,输入寄存器 "x" 宽 22 bit,β = 1。
- 每个 x 模拟 64 shots(共 200,540,928 shots),必须 **0 失败**,容差 `tol(x) = 2⁻²⁴·β/√x` turns(64-bit 定点相位)。
- 评分: `S = Q × C`,Q = 总 qubit 数,C = 每 shot 平均 Toffoli 等效成本(Toffoli + T/4;Clifford Z_POW 免费,细角度 Z_POW 按 `0.56·log₂(1/ϵ) + 5.3` T 门解析定价)。
- 基线: Q = 260,C = 7,988,S ≈ 2.08×10⁶。
- 参考实现: 归一化(成对前导零移位)→ QROM + 三次多项式初值 → 一步 Newton-Raphson → Z_POW 相位 → 逆向 uncompute。

### Benchmark II — DQI(Reed-Solomon 译码,最优多项式交)
- 参数: (m, n, b, r) = (4095, 70, 12, 2016),F_{2¹²}(本原多项式 α¹²+α⁶+α⁴+α+1,小端多项式基),纠 ℓ = 35 个符号错误。
- 合同: `U_c|s⟩|0⟩_W = (−1)^{c·bin(e(s))}|s⟩|0⟩_W`,输入 = 840-qubit syndrome `s` + 49,140-bit 经典测量记录 `c`(误差向量 e(s) 保密)。
- 测试: 按 |a_e|² 采样误差(权 w 分布由 36×36 三对角矩阵 A 的主特征向量给出),N = 102,400 shots,**允许失败**。
- 评分: `S = R(ε⁺)·Q²·T`(qubit 权重是 Toffoli 的两倍),R(ε) = ln 2 / ln(1/(1/2 + 2√ε)) 为重复开销;ε⁺ ≥ 1/16 不给分;0 失败时 ε⁺ ≈ 8.7×10⁻⁴,R ≈ 1.19。
- 基线: Q = 1,893,T = 5,725,461(EEA 关键方程 + Chien 搜索 + Forney 公式 + Itoh-Tsujii 求逆)。

### Benchmark III — DFTHC(FeMo-co 化学模拟,QROM + Majorana 旋转)
- N = 76 轨道,η = 113 电子,双电子积分张量 DFTHC 分解。
- 用户可优化: 分解秩 (R,B,C)、向量 u^{(ρ)}_b、系数 w^{(ρc)}_b 与 W^{(ρc)}、**角度布尔 oracle 电路 Angles_k**(唯一需要显式优化的量子电路)、Givens 旋转树 T、批大小 J、角度精度 β。
- 接受条件: 双电子 Frobenius 误差 ϵ₂ ≤ 20 mHa、单电子 ϵ₁ ≤ 0.020 Ha、Λ > 49.0 Ha、E_gap > 3.914898 Ha;每个 Angles_k ≤ 128 qubit。
- 评分: `S = λ_eff × T_tot × Q_tot`,λ_eff = √(E_gap(2Λ−E_gap));T_tot = 模拟的 Angles 电路 Toffoli(正向+逆向)+ Outer/Inner/Givens 启发式成本;Q_tot = 持久 qubit + 最大临时工作区。

## 目录结构

```
google-verifiable-quantum-circuit-compilation/
├── README.md                 # 本文件
├── docs/
│   ├── verifiable_circuit_compilation_arxiv.pdf   # 比赛论文
│   └── paper_fulltext.txt    # 提取的论文全文
├── data/                     # Kaggle 比赛数据(需先接受规则)
│   ├── verifiable_circuit_compilation.pdf
│   ├── kickmix/              # kickmix.so (Linux) + LICENSE
│   ├── sample_submission/    # invsqrt.kmb / dqi.kmb / dfthcsos/*
│   └── verifiers/            # 三个 benchmark 的官方验证脚本(可本地跑分!)
├── src/                      # 本地开发代码
└── notebooks/                # 分析与实验
```

## 环境与运行方式

- **kickmix 包是 Linux 预编译 .so**,Windows 原生 Python 无法直接 import。
- 官方验证脚本(`data/verifiers/*/verify*.py`)可以在本地直接运行对提交打分,是开发内循环的核心。
- 选项:
  1. **WSL2**(本机现有 Ubuntu-22.04 / kali-linux 虚拟磁盘已损坏,需重装): `wsl --install -d Ubuntu-24.04`
  2. **Kaggle Notebook**: 比赛环境自带 kickmix,直接在线开发/提交
  3. 从源码编译 kickmix(源码暂未公开,不可行)

## 下一步计划

1. [x] 调研比赛与论文
2. [x] 接受比赛规则,下载全部数据(data/ 已就位)
3. [x] WSL Ubuntu-24.04 已安装(scripts/wsl_setup.sh);kickmix.so 为 Python 3.13 编译,正在装 Miniconda py313 环境
4. [x] Windows 侧 DFTHC 经典评估环境已验证(src/dfthc_eval.py,与官方 README 数值完全吻合)
5. [ ] 用官方验证脚本给 sample_submission 打分,验证 WSL 环境
6. [ ] Benchmark III: 分解优化(b3-opt 队友负责;已发现 λ_eff 下界 46.79,基线 49.30 距下限仅 5%;改进主要靠压缩 T_total×Q_total,候选 (R,B,C)=(16,24,24) 等可降 8–14 倍)
7. [ ] Benchmark I: invsqrt 电路优化(基线 Q=260, C=7,988, S≈2.08e6;主要优化乘法器/多项式/寄存器位宽)
8. [ ] Benchmark II: DQI 译码器优化(基线 Q=1,893, T=5,725,461;EEA/GF(2^12)乘法/求逆优化)
9. [ ] 提交并迭代

## 官方评分与提交(Evaluation 页 + metric 笔记本,已核实)

- **总分 = (∏_b cost_b^ref / cost_b)^{1/3}** —— 三个任务相对改进的几何平均(基线提交 = 1.0)。
- 提交:一个 zip,内含 `invsqrt.kmb/.kmx`、`dqi.kmb/.kmx`、`dfthcsos/`(angle_k_*.km[xb] 从 0 连续编号 + submission.npz)。打分约 45 分钟,每天 5 次提交,deadline 2027-01-20。
- 三个基线实测(WSL 官方验证器): B1 2,076,774;B2 2.44e13;B3 3,355,599,588。
- metric 实现: https://www.kaggle.com/code/metric/google-quantum-ai-metric (本地: data/metric/metric.py)
- 附加奖: 里程碑奖 11-20(需公开 notebook)、最长领先奖、三个单项奖各 $10k。

## 目标校准(基于公开参考解的本地官方实测,2026-10-27)

下载并本地验证了公开资源(data/community/):
- **19.54 分团队**完整提交包(vqcc_19_53750,zongaobian 数据集):实质测 **B1 = 108,560(19.1×)/ B2 = 5.05e11(48.4×)/ B3 = 4.08e8(8.2×)**,几何平均 19.7。
- **VQC Verified Optimization Atlas**(kozykappa 数据集):B1 91Q/1,584T、B2 1818Q/583,505T、B3 681Q/72,514T(portfolio 分数 ~5.66);含 `code/compact_horner.py`(signed-digit 参考实现)、banked QROM、33 组精度实验记录。
- **Gu et al. 官方仓库**(andigu/uncompute,Apache-2.0):BM 译码器目标 442,397 T / 1,837 Q。

| 任务 | 基线 | 公开最好(实测) | 我们当前 | 下一步 |
|---|---|---|---|---|
| B1 invsqrt | 2,076,774 | 108,560(19.1×,63Q/1,404T,signed-digit) | 1,627,018(1.28×) | b1-v3 signed-digit 重构 |
| B2 DQI | 2.44e13 | 5.05e11(48.4×,888Q/533kT,BM+条件路由) | 基线 | b2-dec 自研 BM 译码器 |
| B3 DFTHC | 3.356e9 | 4.08e8(8.2×,R6/B192/C68,brot8) | 基线 | 官方训练器 + B>N 形状 |

注意:公开包只用于**技术参考与目标校准**,提交采用我们自己的独立实现(遵守竞赛原创性与对方 export_scope)。

## 进展(实时)

- ✅ **环境**: WSL Ubuntu-24.04 + Miniconda py313 + kickmix + JAX/optax(官方 DFTHC 训练器可跑);Windows 侧纯 numpy 模型。
- ✅ **提交管道**(本机无法直连 Google 存储,改用内核代理): save_kernel 内联源码 → create_kernel_session 在 Kaggle 网络内构建/打包/提交。基线 1.00000、B1 v2 **1.08475** 均已出分。内核侧 B1 构建器与 B3 angle 构建器(src/b3_angle_build.py,精确复现基线 3,355,599,588.32)已验证。
- ✅ **B1**: **已完成 12×** —— v3 signed-digit 架构(91Q/1584T,worst 0.9628×tol)通过官方 200M shots 验证 score=**173,002.75**,与公开冠军持平;已提交(#57695061,2026-10-27)。v2(1.28×,1,627,018)为过程版本。微调实验(drop{5,6}/F1=18/F2=7)全部失败,确认 (8,19,27,1)-drop5 为该架构族最优。
- 🔄 **B2**: b2-dec 的 BM 译码器——经典模型 0 失败;BM 正/反向 + Ω + √ 已实现;b1-v3 已确认反向分支的栈标签规则正确(`rep = d_rev[7] AND NOT(栈顶 tag == r)`,280k 轮 0 错配,见 results/b2_bm_reverse_notes.md);caps=2(溢出率 1.5e-4,总 Q≈1842);剩余:EEA+扫描(Ψ/半次 Bezout 路线,每点 23 T)+ 3 个脏 temp 修复。
- ❄️ **B3**: QROM 侧被证伪(QR 公式对应 QROAM 超 128 限;banked 无净收益;随机 delta 使 segment-tree 近最优,见 results/b3qrom_findings.md);分解侧 (65,60,30) 的 E_gap 下界 763(H0 覆盖不足),优化器守不住 e2=0.0173 初值。兜底 3.3556e9。
- 📊 **排行榜**: **当前 2.28970**(B1 v3);榜首 24.53(10-27)。提交配额 10-27 已用 4/5,保留 1 次给 B2 组合提交(预期几何平均 ~8.3)。

## 提交管道速查

```powershell
python src\gen_submit_kernel.py --message "..."    # 生成内核(内嵌构建源码)
python src\kaggle_kernel.py "zikuanqi/submit-helper" "kernels\submit_helper\submit_helper.py" --run
# 若 409 冲突: python src\kaggle_kernel_rerun.py "zikuanqi/submit-helper" <session_id>
python src\kaggle_kernel_log.py "zikuanqi/submit-helper"    # 查看内核日志
```

## 仓库说明与数据获取

本仓库只包含**代码、文档、分析记录与小体积结果**(data/、submissions/、大 .npz/.kmb 均已 gitignore)。复现需重新获取以下数据(均公开):

```powershell
# 1) 比赛数据(论文 PDF、kickmix.so、sample_submission、verifiers)
kaggle competitions download -c google-verifiable-quantum-circuit-compilation -p data
# 2) 官方 starter kit(dfthc 训练库 + 基线)
kaggle datasets download -d sohier/verifiable-circuit-compilation-starter-kit -p data/starter_kit
# 3) 公开参考(技术参考与目标校准;可选)
kaggle datasets download -d zongaobian/quantum-circuits-signed-digits-bmpsi-and-qrom -p data/community
kaggle datasets download -d kozykappa/quantum-ai-dataset -p data/community
```

环境:WSL2 Ubuntu-24.04 + conda-forge Python 3.13(`numpy scipy jax optax`)+ 比赛数据里的 `kickmix.so`(Linux,Python 3.13 ABI)。脚本前缀说明:`src/b1_*`/`src/b1v3_*` = Benchmark I,`src/b2_*` = Benchmark II,`src/b3_*`/`src/dfthc_*` = Benchmark III,`src/kaggle_*` = 提交基础设施。WSL 相关脚本在 `scripts/`。
