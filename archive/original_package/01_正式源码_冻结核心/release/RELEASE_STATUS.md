# 本次正式封存的测试记录

2026-11-16。原冻结核心的 `dno.py` 完全未修改。原冻结包内29份文件重新比对后逐字节一致；新加的 `examples/` 和接口测试是独立的外部数据读取示例。`release/checkpoint_fresh_results.json` 是本次新采样重新评估的记录；原来的 `evidence/checkpoint_verification.json` 保持归档原样。

- 核心 `tests/test_final.py`：32项通过。
- 接口 `tests/test_modal_interfaces.py`：16项通过。
- 整体：**48项通过**，当前CPU运行约1.34秒，不代表训练时间或长期稳定性。
- 先前保存的4个合成训练检查点重新加载；两份寻址模型的独立新采样MSE约0.000289/0.000279，来源与目标各100%；两份一位历史模型的新采样正确率100%，删去第一输入后分别51.27%与51.86%。
- 用**同一随机未训练权重**处理人造图像patch、视频块、频谱帧、符号向量。四种均能够执行状态更新与观察；**未进行**真实模态识别/生成训练或基线性能对比。
- `release/FILE_MANIFEST.json` 是本次全部正式文件的SHA-256完整性清单；执行 `python release/verify_release.py` 验证。该清单不会包括自身及检验程序输出，避免循环哈希。

**可复现命令**：

```sh
python -m pytest -q tests/test_final.py tests/test_modal_interfaces.py
python verify_checkpoints.py
python examples/run_units.py --out evidence/multimodal_demo.json
python release/verify_release.py
```

以上不意味着DNO已经达到企业生产系统、真实LLM、图像理解、语音识别或视频生成性能。严格证据边界见 `docs/`。
