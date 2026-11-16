# Delta 数学概念与 DNO 神经计算实现：严格定义（研究 v0）

**状态：研究设计及已执行的有限原型。** 本文明确区分：基础概念（不依赖计算机）、建模选择（可以有别的实现）、当前代码事实（可复核）、尚未证明的能力。

## I. Delta：跨领域概念层（不等于神经网络）

下列符号属于**形式化候选**，而非已经证明的普适自然定律。当前状态是过去作用的有效结果，不是完整历史记录，也不要求每一次过去事件都能恢复。

### 1. 独立有效单元（Unit）

存在一组可识别且能独立变化的单元 \(\mathcal U_t\)。单元可具有不同边界与时间尺度，允许组成、分解、消散。独立是拥有局部演化自由度，**不意味着不受环境影响**。

\[u_i(t+1)\in F_i(u_i(t); c_i(t),e_i(t))\]

### 2. 当前有效状态（Present state）

\[\Omega_t=(\mathcal U_t,\mathcal R_t,\mathcal C_t)\]

\(\Omega_t\) 是本时刻有效的全部系统状态（相对于所选描述粒度）。**不是**所有过往事件的存档，也不要求对过去的可逆重构。多个过去序列可映射到同一状态。

### 3. 相互作用（Interaction）

\[\Omega_t \xrightarrow{\iota_t} \Omega_{t+1},\qquad \iota_t\in\operatorname{Enabled}(\Omega_t;\mathcal B)\]

\(\mathcal B\) 是有限底层规律；\(\operatorname{Enabled}\) 受当前状态、关系和边界影响。既可以单元内在演化，也可以单元之间相互作用。

### 4. 关系是有效结构而非纯粹相似性（Relation）

\[r_{ij}(t)\in\mathcal R_t\]

两者即使不相似，也可能发生可行的互补作用；关系可承载自身当前状态并影响未来的作用条件。

### 5. 约束与边界（Constraint and boundary）

\[c_i(t)\in\mathcal C_t,\quad c_i(t+1)=G_i(c_i(t),\Omega_t,\iota_t)\]

内部可演化边界与外部不可越过的实际权限/硬件/数学约束严格区分。不能把所有约束都交由网络自改。

### 6. 可变接合（Conditional encounter）

\[\operatorname{Encounter}_{ij}(\Omega_t)\in\{0,1\}\ \text{或}\ [0,1]\]

发生作用不要求共享词义、固定图边、全局成功分数；但**必须存在具体可执行的候选接触发现机制**。

### 7. 局部自主变化与耦合变化（Intrinsic + coupled change）

\[\Delta u_i = \Delta_{\rm intrinsic}u_i \oplus \Delta_{\rm contact}u_i\]

\(\oplus\) 不要求为普通加法；在具体 v0 代码中使用有界残差和混合更新。

### 8. 关系再作用（Relations act on relations）

\[r_{ij}(t+1)=F_r(r_{ij}(t),r_{ik}(t),r_{jk}(t),u_i,u_j,\ldots)\]

关系间的作用本身也受接合条件约束。这里不预设所有三角关系必须存在。

### 9. 形成、保持、消散（Formation, persistence, dissolution）

\[\mathcal R_{t+1}\ne\mathcal R_t\]\[\mathcal U_{t+1}\ne\mathcal U_t\quad\text{（可选）}\]

并非每个时间步都必须产生新关系，停止和恢复都允许。实际 v0 使用固定单元槽位，只有活跃候选关系与其状态变化；**尚不实现动态新增/删除单元**。

### 10. 非预设的目标中立性（No compulsory global success direction）

Delta 核心变化不必由一个全局奖励最大化法则驱动。**训练神经网络时仍可使用外部损失**，例如预测下一个 Token；这不是将奖励写成系统内一切变化的自然规律。

### 11. 观察与可区分性（Observation）

\[y_t=\mathcal O(\Omega_t),\quad \mathcal O\;\text{可能遗漏内部差异}\]

不要将一次输出相同视为内部状态等价。比较状态还须采用干预/后续可达性实验。

### 12. 可达性与反事实作用（Reachability and intervention）

\[\operatorname{Reach}_H(\Omega\mid r)\overset{?}{\ne}\operatorname{Reach}_H(\Omega\mid \operatorname{do}(\neg r))\]

此处是**实验命题**，不是已证明所有新关系有用。

### 13. 有限规律与开放结果（Finite rules, open outcomes）

\(\mathcal B\) 可有限，产生的状态结构可以十分复杂；但有限资源下不等于无限可计算分支，不自动推出新数学运算原语。

### 14. 有效状态不等于无损历史（State sufficiency is task-relative）

\[\Omega_t=\Phi(x_{\le t})\not\Rightarrow \Phi^{-1}(\Omega_t)=x_{\le t}\]

当前有效状态可能由许多历史共同形成；有限容量可能丢失未来任务需要的细节。这是明确边界。

## II. DNO：仅提取 Delta 在 LLM 神经计算位置上可实现的部分

以下为**具体工程实例，不是 Delta 理论的唯一实现**。DNO 是可微神经网络骨干；语言目标仍是下一 Token 预测。\(\theta\) 为学习的有限共享权重；\(\Omega_t\) 为每个输入序列自己演化的当前状态。两者不同。

### 15. 神经元/单元状态、关系与约束

固定 \(N\) 个槽位（第一版只固定**容量**，槽中状态持续变化）：

\[h_i^t\in\mathbb R^d,\quad c_i^t\in\mathbb R^q,\quad a_i^t\in[0,1]\]

每个活动接触端口保存：

\[e_{i,k}^t\in\mathbb R^r,\quad n_{i,k}^t\in\{-1,0,\ldots,N-1\}\]

其中 \(n_{i,k}\) 是**当前邻居身份**，不是曾经接触过谁的事件日志。若邻居变了且旧关系不在当前候选端口，新端口的关系状态设零；旧状态释放。容量是 \(NK\)，不是 \(N^2\) 全图。

### 16. 输入注入

\[\hat h_i^t=h_i^t+\mathbf 1[i=t\bmod N]\,W_{\rm in}\operatorname{Embed}(x_t)\]

这是首版的轮换槽位注入法，容易复现，但可能造成长期信息遗忘。并非 Delta 概念要求必须这么设计。

### 17. 个体内在变化

\[d_i^t=f_{\rm intrinsic}(\hat h_i^t,c_i^t;\theta)\]

即使没有任何相互作用，仍存在独立的候选变化；是否保持取决于实际更新门。

### 18. 关系发现：有限接触机会而非全注意力

\[\mathcal P_i^t=\underbrace{\{i-1,i+1\}}_{\text{结构局部}}\cup\underbrace{\operatorname{sortNeighbor}_{k_\theta(h,c)}(i)}_{\text{内容索引左右相邻}}\cup\underbrace{\{\operatorname{probe}(i,t)\}}_{\text{有限探索}}\]

使用可学习标量索引 \(k_\theta\)，为每个槽排序，取排序后的前后邻居；时间轮换的探测候选提供开放接触机会。**候选发现是非可微的硬排序**，但索引分数还进入下面的连续接合门，因此索引权重仍可经该路径获得梯度。此法不保证找到所有远距离互补关系。重复端口与自环被排除。

### 19. 当前接合与作用消息

对候选 \(j\in\mathcal P_i^t\)，用当前状态与关系状态计算：

\[z_{ij}^t=[\hat h_i,\hat h_j,c_i,c_j,e_{ij},k_i,k_j,k_i-k_j]\]
\[g_{ij}^t=\sigma(f_g(z_{ij}^t))\]
\[m_{ij}^t=g_{ij}^t\tanh(f_m(z_{ij}^t))\]

不使用 \(QK^\top\) 全量注意力矩阵，也不对所有源节点做 Softmax。候选之间的归一化使用端口数量 \(K\)，而非竞争型注意力权重。关系强弱来自当前局部状态，**不是语义相似度的替身**。

### 20. 关系自身变化与关系再作用

\[\bar e_i^t=K^{-1}\sum_{j\in\mathcal P_i^t}e_{ij}^t\]
\[e_{ij}^{t+1}=\tanh\bigl(0.97e_{ij}^t+0.18 g_{ij}^t f_r(z_{ij}^t,\bar e_i^t,\bar e_j^t)\bigr)\]

函数 \(f_r\) 接受端点处其他关系的当前摘要；因此关系会改变关系。但这是**共享端点关系摘要耦合的一种有限实现**，不是任意高阶超图操作。新关系的状态从零开始；已有且身份不变的关系延续自身状态。

### 21. 边界、个体与活动的同时更新

\[\bar m_i^t=K^{-1}\sum_jm_{ij}^t,\quad\tilde e_i^{t+1}=K^{-1}\sum_j e_{ij}^{t+1}\]
\[c_i^{t+1}=\tanh(0.985c_i^t+0.09 f_c(\hat h_i,c_i,\bar m_i,\tilde e_i^{t+1}))\]
\[v_i^t=\tanh(\hat h_i+0.12d_i+0.25\bar m_i)\]
\[h_i^{t+1}=(1-0.22\gamma_i)\hat h_i+0.22\gamma_i v_i^t,\quad\gamma_i=\sigma(f_u(\hat h_i,c_i,\bar m_i))\]
\[a_i^{t+1}=0.90a_i^t+0.10\sigma(f_a(h_i^{t+1},c_i^{t+1},a_i^t))\]

系数是**明确标出的实验常数**，没有理论保证为最优。局部边界可变化；外部执行权限不在本算子内改变。

### 22. 语言模型读出和训练

\[z_t=\operatorname{LN}([h_{t\bmod N}^{t+1},\sum_i a_i^{t+1}h_i^{t+1}/(\sum_i a_i^{t+1}+\epsilon)])\]
\[p_\theta(x_{t+1}\mid x_{\le t})=\operatorname{softmax}(W_o z_t)\]
\[\mathcal L=-\sum_t\log p_\theta(x_{t+1}\mid x_{\le t})\]

通过 BPTT/AdamW 训练共享权重；**不需要推理时重放所有历史输入**。但训练展开序列需要保存自动微分图，这是训练技术所需，不是推理状态中保存全部历史。

## III. 实际代码与理论的对应关系

| 基础概念 | DNO 工程实现 | 已实现的边界 |
|---|---|---|
| 个体自主性 | `autonomous` + `unit_gate` | 固定数量槽位，共享权重 |
| 当前状态 | `DNOState.h`, `.boundary`, `.relations`, `.activity` | 有损有限容量 |
| 关系发现 | `candidate_neighbors` | 5 个端口；硬排序不保证全局最优 |
| 条件性接合 | `pair_gate` | 连续可微门，关系候选离散 |
| 作用 | `message` | 共享神经映射，不是通用新数学操作 |
| 关系变化 | `relation_update` | 同一端口身份延续，变更则释放 |
| 关系再作用 | `old_summary`, `other_context` | 汇总近邻的有限高阶作用 |
| 可变边界 | `boundary_update` | 内部数值边界；不改变系统权限 |
| 保持与消散 | 衰减、有界更新、端口替换 | 尚无任意单元生成/销毁 |
| 当前读出 | `readout` | 预测 Token；没有自然语言理解证据 |

## IV. 复杂度与不夸大的保证

对批次 \(B\)、槽位 \(N\)、端口 \(K=5\)、长度 \(T\)：

- 候选发现：每步学习索引与排序，大致 \(O(BN\log N)\)，局部/探测构建 \(O(BNK)\)。
- 消息和关系计算：依赖网络维度，约 \(O(BNK(d+q+r)d)\)，**不是**简单固定为 \(O(BNK)\) FLOPs。
- 当前持久张量：\(O(BN(d+q+Kr+K))\)，不存 \(T\) 个历史 Token。
- 训练时 BPTT 会保留时间展开的中间激活，内存随 \(T\) 增长。
- 代码在 `_gather_units` 中使用 `expand` 视图，逻辑形状可含 \(N\times N\) 的**零拷贝视图**，未物化完整 pairwise-score 矩阵；未来的真实 GPU profiling 仍必须检验峰值内存与吞吐。
- 固定 \(N\) 个槽位使长期上下文能力受限制。\(N\) 不随着序列增长不代表信息能无损保留。

## V. 明确不成立的推论

1. **能训练**不等于**已经形成自然语言理解**。
2. **出现新关系状态**不等于**发现任意数学新运算**。
3. **候选少于全连接**不等于**实际能耗一定低**。
4. **某个实验更准**不等于**优于 Transformer/Mamba**。
5. **状态凝聚既有作用**不等于**每段历史都可重建**。
6. **有限测试中没有发散**不等于**无限时间稳定**。
7. **关系影响关系的代码存在**不等于**这一机制已证明在语言任务中不可替代**。

## VI. 参考资料（用于区别已有研究，不构成 Delta 原理的来源证明）

- Vaswani et al., Attention Is All You Need, 2017/2023: https://arxiv.org/abs/1706.03762
- Gu & Dao, Mamba: Linear-Time Sequence Modeling with Selective State Spaces, 2023: https://arxiv.org/abs/2312.00752
- PyTorch `gather`: https://docs.pytorch.org/docs/stable/generated/torch.gather.html
- PyTorch `argsort`: https://docs.pytorch.org/docs/stable/generated/torch.argsort.html

**结果结论**：DNO v0 是一套从 Delta 概念提取的、已能反向传播训练的候选神经算子；可否成为有竞争力的 LLM 底层骨干，尚未证明。
