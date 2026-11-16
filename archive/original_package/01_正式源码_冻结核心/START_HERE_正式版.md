# Delta / DNO 正式版：核心不变，多坐标观察独立

**归档日期：2026-11-16。性质：正式研究代码发行，不是另一版生成算法。**

## 最重要的声明

此发行的唯一生成核心是根目录的 `dno.py`，它与此前封版源码**逐字节相同**：

`SHA256(dno.py) = fea8d9ea81c94dbf926b984b3c158f301afc71782f989c73b3c9bd02fac767b1`

不是“观察不够时再增加内核组件”的设计。**目前尚未识别的东西，先检查观察条件、尺度、数据与表示方式；仍看不见时允许暂不解释。** 这不代表永远否认核心可能存在真正缺陷。

`multiview_observer.py`、`extended_readonly_views.py` 和 `stress_audit.py` 是**外部分析仪器**，并没有被 `dno.py` 导入；它们也不是模型的数学本体。历史研究的素数频率、动态增维、反事实SVD、规律判定器等都没有并入正式核心。

## 文件地图

- `dno.py`：冻结的唯一 DNO 生成动力学。`frozen/dno.py` 是一模一样的审计副本，服务此前研究测试导入，不是第二个实现。
- `MATH_FINAL_ZH.md`、`FROZEN_SCOPE_ZH.md`、`FINAL_REPORT_ZH.md`：原始封版数学契约、设计边界和当时实测。
- `docs/01_完整研究记录_问题尝试训练结论.md`：v0–v5、封版、升维、素数视角、反事实与合成训练的原有详细记录。
- `docs/02_数学理解与代码对应.md`：抽象与实际有限代码的关系。
- `docs/03_图像视频声音_单元表示预测与检验.md`：跨模态预测与严格限定。
- `docs/04_多坐标观察与未定判断.md`：这次确定的坐标分析方法；不把观察结论写回核心。
- `docs/05_完整测试证据及边界.md`：本轮重新执行的结果、曾经的观察盲点、负面反例。
- `research_archive/`：多视角完整审计的原研究报告（归档）。
- `multiview_observer.py`：原独立观察器，保留其身份漏检反例供研究。
- `extended_readonly_views.py`：可选的身份、主动寻址、有限时间读数，**仅外部分析**。
- `stress_audit.py`：160随机初始化的观察压力测试。
- `tests/test_final.py`：原核心32项；`tests/test_modal_interfaces.py`：跨模态数值接口16项；`research_tests/`：多视角研究114项。
- `evidence/`：四份原训练检查点、历史实验输出与这次重新采样的审计结果。
- `release/`：哈希、完整性记录、运行日志。
- `examples/coordinate_demo.py`：只读观察示范。
- `LICENSE`、`NOTICE`：先前约定的 Apache-2.0 与作者元数据。

## 运行要求

Python 3.10+，PyTorch、numpy、pytest（`requirements.txt` 保留原需求；numpy 是外部多坐标研究工具的额外依赖）。

在该文件所在目录运行：

```bash
python -m pytest -q tests/test_final.py tests/test_modal_interfaces.py research_tests
python verify_checkpoints.py
python stress_audit.py --seeds 160 --output evidence/multiview/stress_independent.json
python examples/coordinate_demo.py
python release/verify_release.py
python release/verify_package.py
```

可从根目录直接使用核心：

```python
import torch
from dno import DNO
m = DNO(d=8, hidden=24)
s = m.introduce(m.blank(), torch.randn(2, 8))
s = m.evolve(s, 0.1)  # 自身演化；不强制发生组合或观察
s.validate()
print(s.ids, s.value.shape)
```

## 证据与解释的边界

- 原核心的32项、跨模态输入合同16项和多视角观察114项重新执行，共162项通过；160个种子包含1,280组指定扰动正例，其中旧观察器故意保留的身份盲点导致160次未检出。**检测工具有盲点**，不是内核失败。
- 四份既有小型合成任务检查点可重新加载并在新采样上通过；**没有**从本发行得到新基础模型、新的真实图片/语音/视频能力或普适规律证明。
- 相同的数值观测不能推出全部等价；视角切换也不是万用解释器。发现缺陷需要独立任务、可重复失败和适当对照。

## 最终约定

**不修改已经验证的 DNO 核心，不为每个数学解释添加一个内部模块。** 多坐标观察、切换、不同时间尺度、证据不足和暂不解释是外部研究方法。只有明确可重复的核心行为失败，才构成进一步研究模型缺陷的理由。这里并不宣布任何新的底层数学定律。
