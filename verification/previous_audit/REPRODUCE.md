# 本次重新执行的证据与复现指令

操作日期：2026-11-16。Python 3.13.5，torch 2.10.0+cpu，pytest 9.0.2，numpy 2.3.5。

实际重新执行：

```bash
cd 01_正式源码_冻结核心
python -m pytest -q tests/test_final.py tests/test_modal_interfaces.py research_tests
python release/verify_release.py
python release/verify_package.py
python stress_audit.py --seeds 160 --output /tmp/dno_stress.json
cd ..
python 04_重测证据/verify_checkpoints_readonly.py
python 04_重测证据/verify_candidate.py
```

- 原测试在本会话中重新执行：162/162通过。
- 原发行版哈希核验：44个记录项通过，冻结核心一致。
- 原72文件完整性核验：71项依原发布核验逻辑通过，冻结核心一致（不计被检验清单自身的哈希循环）。
- 4个已有训练检查点重新加载并用生成的新样本复测：4/4通过；记录在 `checkpoint_重新执行.json`。
- 160随机初始化、1,280正例及320负例观察复测：详见 `stress_160_重新执行.json`。其中旧观察器160组身份标签漏检是**被保留、准确检出的观察工具边界**。

原始的全部已发行日志、权重和合成训练数据位于 `../01_正式源码_冻结核心/evidence/` 和 `release/`。

这些测试属于规定的有限验证集；任何后续更大训练、模型对照及新涌现效果，都需要另附新任务与实测证据。
