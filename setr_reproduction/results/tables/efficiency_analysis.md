# 环节 12：效率分析结果

## 1. 效率主表

| method_label | answer_f1 | avg_selected_passages | avg_context_tokens_est | token_reduction_vs_retrieval_top5 | avg_generation_latency_seconds | avg_total_latency_seconds | answer_f1_per_1k_tokens_est |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Retrieval Top-5 | 0.5931 | 5.0 | 531.9705 | 0.0 | 0.4348 | 0.4348 | 1.1149 |
| BGE-Reranker Top-5 | 0.6297 | 5.0 | 581.0935 | -0.0923 | 0.4438 | 6.6225 | 1.0836 |
| LLM Listwise Top-5 | 0.6477 | 5.0 | 568.1175 | -0.0679 | 0.4468 | 2.5088 | 1.1401 |
| SetR-Selection only | 0.6602 | 2.308 | 251.793 | 0.5267 | 0.3401 | 1.3688 | 2.622 |
| SetR-CoT | 0.6496 | 2.266 | 249.4825 | 0.531 | 0.341 | 2.463 | 2.6038 |
| SetR-CoT + IRI | 0.6582 | 2.224 | 245.1155 | 0.5392 | 0.3382 | 4.9371 | 2.6853 |

## 2. 验证结果

- Top-5 baseline 平均 passage 数校验：True
- SetR 平均 passage 数校验：True
- passage 数与估算 token 数 Pearson 相关系数：0.9956
- warnings：[]

## 3. 关键结论

SetR-Selection only 平均选择 2.308 个 passage，估算上下文 token 为 251.793，相比 Retrieval Top-5 token 减少 0.5267。

SetR-CoT 平均选择 2.266 个 passage，估算上下文 token 为 249.4825，相比 Retrieval Top-5 token 减少 0.531。

SetR-CoT + IRI 平均选择 2.224 个 passage，估算上下文 token 为 245.1155，相比 Retrieval Top-5 token 减少 0.5392。

SetR 系列显著减少输入上下文长度，且答案 F1 与 Top-5 baseline 相当或更高，说明集合选择在本实验中具备较好的上下文压缩效率。

需要注意：BGE-Reranker 的 selection latency 明显高于其他方法；LLM/SetR 方法的 latency 来自远程 Qwen3-14B API，受服务端负载影响，报告中应把延迟结果作为实验环境下的观测值，而不是模型绝对性能结论。
