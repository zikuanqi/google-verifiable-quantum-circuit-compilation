# B3 ANGLE QROM 优化研究(90 分钟时间盒)— 结论与证据

## 核心结论

**对 baseline 分解的随机性 angle_words,在 128-qubit 合同内,T ≤ 50,000 不可达;
QR 公式对应的 QROAM 结构超出合同硬上限。** 建议 QROM 优化与 b3-opt 的新分解
(结构化 words)联合设计;banked 结构本身对 baseline 数据无净收益。

## 证据链

1. **合同硬约束**(SCORING_README): 每个 ANGLE_k 电路 ≤ 128 qubits(QUBIT_CAP,
超限直接 reject);selection 恰好 12 位;word_j 12 位(brot=12);无 Z_POW;
workspace 归零。基线: A=3976 行, J=8, M1=96 位, 10 批 × 2 passes,
naive segment-tree = A−2 = 3,974 Toffoli/pass, ANGLES = 79,480,
T_total = 93,237。

2. **QR(L,M1)=L/2^k+M1·(2^k−1) 的结构 = Babbush/Qualtran QROAM(swap registers)**:
k=3 需要 2^k·M1 = 768 个 swap qubit(另有 selection 12 + words 96 + anc):
~890 qubits ≫ 128,被合同直接拒绝。k=1(k=1: 2^1·96=192)同样超限。
→ QR 公式定价的结构无法进入合同。

3. **2-地址 banked XOR QROM(实测,rank-reduced 差分库)**:
   - J=8: 198 qubits / 2,154 Toffoli/pass(超限 70)
   - J=5: 132 qubits / 2,124/pass(超限 4;差分库 GF(2) rank=50, 无法压进 45)
   - J=4: 110 qubits / 2,114/pass ✓ 合限,但 19 批 × 2 × 2,114 = 80,332 ≈ 基线 79,480,无收益
   (walk 1986 AND + mux W CCX + del_lookup ~70 是硬成本;GF(2) 列秩 5W/6,
   缺口来自 one-body 行的结构,不是可分压缩)。

4. **baseline delta 列的 ANF 复杂度(全 4096 行 Mobius 变换)**:
   每列 0..2120 项(43 列 degree 12、37 列 degree 11),96 列用满全部 4096 个
   不同 support——即 words 是地址的密码学式随机函数。任何 XOR-AND DAG 合成
   对随机函数需要指数级 AND;segment-tree(3974)对随机数据已接近最优。

5. **19.5 团队 337-Toffoli 电路的真相**:
   - 他们的分解是重构过的: R,B,C = (6,192,76)(基线是 (65,60,60)),A=1228,
     brot=8, J=5——问题形状完全不同;
   - 电路 = XOR-AND DAG 合成(337 CCX + 43,922 免费 CX 输出 XOR + 71 HMR+CZ 修复),
     是 MrMeows 系的「shared XOR-AND DAG + 前缀特征」编译器产物,配套 SAT 证书;
   - 我恢复并 ANF 分析了他们自己的 delta1: 也是全复杂度(1076 项/列,2048/2048
     supports)——说明其合成利用了更深的机制(前缀分解/自由行利用),90 分钟内
     未能完全逆向;但可确定它绑定在他们重构过的数据上。

## 产出文件

- scripts/b3qrom_probe.py / b3qrom_banked.py / b3qrom_anf.py / b3qrom_recover.py
  (rank 探测、banked 实测、ANF 分析、19.5 团队 delta 恢复分析)

## 建议

1. **短期无交付**(不满足 ≤50k 目标,按时间盒停止)。
2. 若 b3-opt 的新分解能产出结构化 words(如 19.5 团队那样重构 R/B/C 形状 +
   低 brot),再联合做 XOR-AND DAG 合成(参考 MrMeows dfthcsos_8p5558 编译器
   + certificates),目标才是 23k-37k T。
3. 独立于分解的一个小项: J=4 banked(110Q/2114T/pass)≈ 基线,不值得单独提交。
