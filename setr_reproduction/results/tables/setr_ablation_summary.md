# 环节 11：SetR 消融分析结果

## 1. 总体结果

| method_label | answer_em | answer_f1 | evidence_recall | evidence_precision | all_support_hit | avg_selected_passages | avg_context_words | avg_total_latency_seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SetR-Selection only | 0.508 | 0.6602 | 0.855 | 0.8279 | 0.724 | 2.308 | 165.376 | 1.3688 |
| SetR-CoT | 0.502 | 0.6496 | 0.858 | 0.8349 | 0.732 | 2.266 | 163.722 | 2.463 |
| SetR-CoT + IRI | 0.506 | 0.6582 | 0.834 | 0.8391 | 0.678 | 2.224 | 160.936 | 4.9371 |

## 2. 增量变化

| comparison_label | delta_answer_em | delta_answer_f1 | delta_evidence_recall | delta_evidence_precision | delta_all_support_hit | delta_avg_selected_passages | delta_avg_total_latency_seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SetR-CoT - SetR-Selection only | -0.006 | -0.0106 | 0.003 | 0.007 | 0.008 | -0.042 | 1.0942 |
| SetR-CoT + IRI - SetR-CoT | 0.004 | 0.0086 | -0.024 | 0.0042 | -0.054 | -0.042 | 2.4741 |
| SetR-CoT + IRI - SetR-Selection only | -0.002 | -0.002 | -0.021 | 0.0112 | -0.046 | -0.084 | 3.5683 |

## 3. 样本级变化统计

| case_tag | count | ratio |
|---|---:|---:|
| cot_helped_answer_em | 5 | 0.0100 |
| cot_helped_evidence | 18 | 0.0360 |
| cot_hurt_answer_em | 8 | 0.0160 |
| cot_hurt_evidence | 14 | 0.0280 |
| iri_helped_answer_em | 7 | 0.0140 |
| iri_helped_evidence | 17 | 0.0340 |
| iri_hurt_answer_em | 5 | 0.0100 |
| iri_hurt_evidence | 44 | 0.0880 |
| same_or_minor_change | 411 | 0.8220 |

## 4. 阶段结论

CoT 相比 Selection only 的 F1 变化为 -0.0106，Evidence Recall 变化为 0.003，All-support Hit 变化为 0.008。

IRI 相比 CoT 的 F1 变化为 0.0086，Evidence Precision 变化为 0.0042，All-support Hit 变化为 -0.054。

在当前 prompt-level 复现设置下，IRI 的主要影响是进一步减少平均选择 passage 数，并略微提高 evidence precision；但它降低了 evidence recall 和 all-support hit，说明显式信息需求识别在没有微调模型配合时可能使选择策略过于保守。

三个 SetR 变体的最终答案质量差距很小，其中 SetR-Selection only 的 EM/F1 最高；因此报告中应把 IRI 写成“高精度但可能漏证据”的消融现象，而不是强行写成稳定正收益。

## 5. 关键数值

- SetR-Selection only: EM=0.508, F1=0.6602, All-support Hit=0.724, Avg Selected=2.308
- SetR-CoT: EM=0.502, F1=0.6496, All-support Hit=0.732, Avg Selected=2.266
- SetR-CoT + IRI: EM=0.506, F1=0.6582, All-support Hit=0.678, Avg Selected=2.224
