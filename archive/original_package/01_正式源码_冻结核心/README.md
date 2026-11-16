# DNO — Delta 最终压缩研究核心

**本交付是冻结的、可独立运行的最终研究实现；不是大型语言模型，也不是 Delta 数学已被证明。** 本次不继续扩展版本、不发起下一轮提问。

## 核心结构

`dno.py` 是唯一底层神经核心。没有固定事件种类表、全局有向单元对枚举、强制时间/事件调度器、历史记录库、专用记忆模块或任务语言词汇表。

每个当前有效单元、每个因作用而产生的关系/组合，都使用同一种承载体 `(value, boundary, law, support, age, id)`。当前连接只是稳定身份之间的结构。一个共享 `generator` 用于自主局部演化和二元实际相互作用（后者可能留下可再次参与作用的第三承载体）。

- `blank`, `introduce`：开始及外部新单元出现；不规定它必须是什么领域对象。
- `evolve`：即使没有新外部输入也能递归；局部提示只作用于指定身份；**不自动读取全体关系向邻居发送消息**。
- `attend`、`observe`：主动观察是可选的，平常 `evolve` 不会偷读所有承载体形成观察。
- `propose`：从当前有效状态按需展开一次双端地址和活动支撑；内部没有先构造全部单元对。
- `compose`：同一个共享生成规律改变两端，按正支撑生成一个可再组合的当前承载体；**何时执行由应用或调用方决定**，不强制某个固定事件时钟。
- `compact`：仅释放确切零支撑，旧行为影响留在当下状态而非独立事件日志。

`MATH_FINAL_ZH.md` 是最小数学表达和最早定义的逐项对应；`FINAL_REPORT_ZH.md` 给出真实测试与尚未实现的理论边界。早期 v0 的数学与代码已复制到只读参考资料，不覆盖原文件。

## 执行方式

```bash
python -m pip install -r requirements.txt
python -m pytest tests/test_final.py -q
python demo.py
python train_synthetic.py --out evidence --pair-steps 450 --history-steps 350
python verify_checkpoints.py
python probe_counterfactuals.py
```

`train_synthetic.py` 有两个普通任务训练（最终数值差与延迟一位分类），**不标注“应接触的单元对”或“记忆锚点”**；并不等于无监督学习。

## 交付文件

- `dno.py`：最终核心。
- `MATH_FINAL_ZH.md`：极简抽象数学。
- `FINAL_REPORT_ZH.md`：全部实测、反例和边界。
- `tests/test_final.py`：数学约束对应测试。
- `train_synthetic.py`、`verify_checkpoints.py`：训练与重载确认。
- `probe_counterfactuals.py`：64组随机反事实。
- `evidence/`：测试日志、训练原始 JSON、四份可重载模型权重、反事实原始结果。
- `reference/`：最早 v0 数学/原始代码的原封存档，以及上一轮数学约束文档。
- `SOURCE_MANIFEST.json`：文件哈希核验。
- `FROZEN_SCOPE_ZH.md`：本次交付冻结约定。

## 权限与许可

原有研究材料独立保留，未向 GitHub 推送。**本交付没有擅自选定开源许可证或声称已公开发布。**
