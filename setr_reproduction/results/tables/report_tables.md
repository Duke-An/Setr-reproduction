# SETR 论文复现报告表格

本文件汇总课程论文中可直接使用的核心表格。所有数值均从实验输出 CSV / JSONL 自动生成，避免手动转录误差。

## 表 1：主实验结果

| Method | EM | F1 | Ans. Contains | Evidence Recall | Evidence Precision | All-support Hit | Avg Selected | Avg Context Words | Total Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Retrieval Top-5 | 0.456 | 0.5931 | 0.54 | 0.859 | 0.3436 | 0.734 | 5 | 348.81 | 0.4348 |
| BGE-Reranker Top-5 | 0.488 | 0.6297 | 0.582 | 0.898 | 0.3592 | 0.8 | 5 | 380.714 | 6.6225 |
| LLM Listwise Top-5 | 0.504 | 0.6477 | 0.592 | 0.893 | 0.3572 | 0.8 | 5 | 372.654 | 2.5088 |
| SetR-Selection only | 0.508 | 0.6602 | 0.602 | 0.855 | 0.8279 | 0.724 | 2.308 | 165.376 | 1.3688 |
| SetR-CoT | 0.502 | 0.6496 | 0.59 | 0.858 | 0.8349 | 0.732 | 2.266 | 163.722 | 2.463 |
| SetR-CoT + IRI | 0.506 | 0.6582 | 0.6 | 0.834 | 0.8391 | 0.678 | 2.224 | 160.936 | 4.9371 |

结论：SetR 系列在平均选择 passage 数显著少于 Top-5 baseline 的情况下，答案 EM/F1 不低于各 baseline；但 BGE/LLM Top-5 的 evidence recall 和 all-support hit 更高，说明 SetR 更偏向高精度、低冗余选择。

## 表 2：SetR 消融实验

| SetR Variant | EM | F1 | Evidence Recall | Evidence Precision | All-support Hit | Avg Selected | Total Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SetR-Selection only | 0.508 | 0.6602 | 0.855 | 0.8279 | 0.724 | 2.308 | 1.3688 |
| SetR-CoT | 0.502 | 0.6496 | 0.858 | 0.8349 | 0.732 | 2.266 | 2.463 |
| SetR-CoT + IRI | 0.506 | 0.6582 | 0.834 | 0.8391 | 0.678 | 2.224 | 4.9371 |

结论：CoT 对证据完整性有轻微提升，但没有稳定提升答案质量；IRI 进一步提高 evidence precision 并压缩上下文，但降低 evidence recall 和 all-support hit，体现出 prompt-level IRI 的保守倾向。

## 表 3：效率分析

| Method | F1 | Avg Selected | Est. Tokens | Token Reduction | Gen. Latency | Total Latency | F1 / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Retrieval Top-5 | 0.5931 | 5 | 531.9705 | 0.00% | 0.4348 | 0.4348 | 1.1149 |
| BGE-Reranker Top-5 | 0.6297 | 5 | 581.0935 | -9.23% | 0.4438 | 6.6225 | 1.0836 |
| LLM Listwise Top-5 | 0.6477 | 5 | 568.1175 | -6.79% | 0.4468 | 2.5088 | 1.1401 |
| SetR-Selection only | 0.6602 | 2.308 | 251.793 | 52.67% | 0.3401 | 1.3688 | 2.622 |
| SetR-CoT | 0.6496 | 2.266 | 249.4825 | 53.10% | 0.341 | 2.463 | 2.6038 |
| SetR-CoT + IRI | 0.6582 | 2.224 | 245.1155 | 53.92% | 0.3382 | 4.9371 | 2.6853 |

结论：SetR 系列将估算上下文 token 数减少约 52.7%–53.9%，同时保持较高答案 F1；其中 SetR-Selection only 的综合效率最好。Token 数为 `avg_context_chars / 4` 的估算值。

## 表 4：案例分析表

| Case Type | Question | Gold Answer | Gold Titles | Top-20 All-support | 观察 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SetR-CoT + IRI 成功，Top-5 / reranker 失败 | How far from Sacramento is the flight school in Atwater? | about 115 miles (185 km) | Sierra Academy of Aeronautics, Castle Air Force Base | 1 | SetR 变体只选择紧凑的 gold evidence 并答对，而 Top-5 类方法虽然包含证据，却输出 Insufficient Information。 |
| reranker 成功，SetR 失败 | In which song was written by singer-songwriter Taylor Swift and shares the optimistic lyrical message to a ... | Shake It Off | Yodel It!, Shake It Off | 1 | LLM Listwise 保留了更宽的上下文并答对；SetR 变体虽然选择了紧凑证据，但生成器仍失败。 |
| 三者都失败，原因是 top-20 候选池没召回完整 gold evidence | What song was number 4 on the charts when a song from FutureSex/LoveSounds was number 1? | Rudebox | Rudebox (song), SexyBack | 0 | Hybrid Top-20 缺少 gold title `Rudebox (song)`，后续 reranking / SetR 无法恢复候选池外的证据。 |
| SetR 选更少 passage 但答案正确 | What edible, juicy fruit is grown on a deciduous tree called 'pesco' in Italian? | peach | Pesco, Peach | 1 | SetR-Selection only 仅使用 2 条 passage（Peach, Pesco）即覆盖完整证据并生成正确答案。 |
| Qwen3-14B 输出格式错误或选择冗余 passage | Which star in the movie Hush was born April 20, 1949? | Jessica Lange | Hush (1998 film), Jessica Lange | 0 | 候选池缺少关键证据，且 prompt-level 选择无法补救，最终生成 Insufficient Information。 |

结论：案例表显示 SetR 的优势主要来自上下文压缩和去冗余；失败边界主要来自 first-stage retrieval 未召回完整证据，以及 prompt-level SetR / generator 对证据利用不稳定。

## 数据来源与产物状态

- main results: `D:\Projects\NLPFinal\setr_reproduction\results\tables\main_results.csv`
- SetR ablation: `D:\Projects\NLPFinal\setr_reproduction\results\tables\setr_ablation.csv`
- efficiency results: `D:\Projects\NLPFinal\setr_reproduction\results\tables\efficiency_results.csv`
- case analysis: `D:\Projects\NLPFinal\setr_reproduction\results\cases\case_analysis_cases.jsonl`
- Excel workbook: not generated; optional output skipped because Node / artifact-tool runtime is unavailable in this workspace
