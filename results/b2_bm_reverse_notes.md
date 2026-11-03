# BM 反向轮分支检测 — 边界情形正确判定规则

(结论: 用 fused-Q 栈标签规则,放弃 τ[0]!=τ[35] 判据;当前 b2_build.py 的代码已是正确
规则,只有模块 docstring(第 10-11 行)还是旧规则,需要更新。)

## 1. 参考实现的规则(circuit.py,Gu et al.)

circuit.py 的 BM 反向不重新"猜测"分支——它用 **fused-Q 移位栈的轮标签**精确判定
(close_ctl / erase 的恒等式):

```
ctl == dW[SGN] AND NOT [top tag == r]
```

即:`rep = d_rev[7] AND NOT (top_tag == r)`,其中:
- `d_rev` = 反向轮开始时 `isub(1)` 之后的计数器(正向轮结束时的补码值:
  rep 轮 `d_rev = ~d_old = -d_old - 1`(符号位 = 1),nrep 轮 `d_rev = d_old`);
- `top_tag` = 栈顶槽的轮标签(正向 push 时写入的 r)。

判定不引用任何 tau/sigma 系数,τ[0]==τ[L] 巧合**根本不会进入判定**。

## 2. τ[0]!=τ[35] 判据为何会漏检(你看到的边界情形)

旧规则 `rep = (tau0 != 0) and (d <= 0) and (tau0 != tau35)` 用正向后的 tau 状态判别:

- **replace 分支漏检**(真 rep 被 guard 杀掉): rep 轮的正向后,路由交换使
  `tau_post = sigma_pre`,于是 `tau0_post == tau35_post  ⟺  sigma0_pre == sigma35_pre`。
  这个巧合(大域 GF(2^12) 概率 2^-12,小域测试 GF(2^4) 概率 2^-4,恰好是你报告的
  两个概率)发生时 guard 把真 rep 判成 nrep。
- **假阳性**(另一侧): nrep 轮的 tau 是循环左移:`tau0_post = tau_pre[35]`
  (wrap),`tau35_post = tau_pre[34]`。当 `tau_pre[35] != 0` 时需要 guard 压掉假阳性;
  guard 本身在 `tau_pre[35] == tau_pre[34] != 0`(同样 ~2^-12)时失效。

即:τ 判据的误差 = 2^-12 × P(轮次落在最高次数 L=35 附近),大域很稀但在小域测试里
≈ 2^-4,属于**结构性漏检**而非可忽略噪声;栈标签规则是精确的,零漏检。

## 3. 正确规则(已验证 280,000 轮 0 错配)

```python
# 正向轮末尾(d 更新,与现代码一致):
#   push: 控制 = d[7] (pre-update 符号位), 即每个 d<0 的轮都 push
#         (含 delta==0 的 d<0 轮,delta 值推 0 亦可);
#   tag:  cx(d[7], stk[0][degree+bp]) for r 的置位 — 未溢出时被腾空槽的 tag 位是 |0>,
#         故 tag == r 精确成立;
#   d 更新: cx(rep, d); iadd(1, d)  =>  rep: d<- -d ; nrep: d<- d+1
#         (two's complement: ~d + 1 = -d)
# 反向轮开头:
d -= 1                        # isub(1, d); 此后 d[7] = rep OR pushed
rt = [stk[0].tag == r]        # flip_if_equal(stk[0][degree:], r, rt)
rep = d[7] AND NOT rt         # 你的现实现: cx(d7,rep); ccx(d7,rt,rep)
```

全边界情形表(d_old = 正向轮 pre-update 计数器,delta = 本轮 discrepancy):

| 情形 | 正向 rep | 正向 push | 反向 d_rev | d_rev[7] | rt | 判定 rep |
|---|---|---|---|---|---|---|
| delta≠0, d_old≥1 | 1 | 0 | -d_old-1 <0 | 1 | 0 | 1 ✓ |
| delta≠0, d_old=0 | 1 | 0 | -1 | 1 | 0 | 1 ✓ (d=0 边界) |
| delta=0, d_old≥0 | 0 | 0 | d_old | 0 | 0 | 0 ✓ |
| delta≠0, d_old<0 | 0 | 1 | d_old | 0 | 1 | 0 ✓ |
| delta=0, d_old<0 | 0 | 1 | d_old | 0 | 1 | 0 ✓ (delta=0 也 push) |
| 栈溢出 (depth>caps) | — | — | — | — | tag 被破坏 | 声明失败(E2 类) |

溢出时被丢弃条目的 tag XOR 进新 tag("rotated-out tag corrupts the compare",
circuit.py 原话),判定不可信 → 归入 declared-failure 预算(需实测 P(depth > caps),
caps=2/3 的溢出率极小)。

## 4. 对 b2 代码的核对结果

- `b2_build.py` 的 `bm_round_rev`(第 219-317 行)+ `BMRegs` docstring(第 96-101 行)
  **已经是精确的栈标签规则**;push 级联、tag 写入、d 补码、`rep = d[7] AND NOT rt`
  均与参考一致。**只需把模块 docstring 第 10-11 行的旧规则文字改成栈标签规则**。
- `b2_model.py` 的 `round_rev`(`rep = (self.d < 0) and not pushed`)同规则,正确。
- 反向轮 m 的重算恒等式(正确路径):rep 轮 `m = sigma0_post * inv(tau0_post)`,
  由 q-form 不变式 `sigma0_post * delta == 1`(σ0·Δ=1)精确等于正向的
  `inv(delta*sigma0_pre)`;nrep 轮 `m = popped_delta * sigma0`。近似路径上不变式
  破坏 → 归 declared-failure(模型已计 fail_stack/fail_cutoff)。
- 我本地验证:GF(2^12)、4,000 样本 × 70 轮 = 280,000 轮:栈标签规则 0 错配;
  τ 规则在大域 0 失效(其失效需要 L=35 顶次数巧合,大域极稀),与小域 2^-4 的报告
  一致(τ 规则只在顶次数巧合时坏,但这正是无法接受的结构性漏洞)。

## 5. 结论(给 b2-dec)

1. **保留现栈标签规则**,不要退回 τ 判据;τ[0]==τ[L] 巧合不是边界 bug 的来源。
2. 更新 `b2_build.py` 第 10-11 行 docstring。
3. 若想进一步省:参考实现里 push 控制是 d<0(**pre-update**),与你的 `bm.d[7]`
   一致,不必改。
4. 唯一需要实测预算的边界 = 栈溢出率(见上表),跑 `run_experiment` 的
   `stack_overflow` 计数确认 < DQI 失败预算。
