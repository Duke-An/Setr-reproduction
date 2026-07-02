# First-stage Retrieval Quality 对比表

评估对象：HotpotQA dev distractor 随机 500 个问题。

| 候选池 | 类型 | 样本数 | Avg Returned | Evidence Recall | All-support Hit | Gold Title Hit | Parse Failed Candidates | Avg Latency(s) | 用途 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Candidate Context | Diagnostic / Oracle-style | 500 | 9.972 | 1.000 | 1.000 | 1.000 | 0 | - | 诊断 SetR / rerank 在 gold evidence 已进入候选池时的选择能力 |
| Dify Vector Top-20 | Global Corpus Retrieval | 500 | 19.890 | 0.906 | 0.816 | 0.996 | 0 | 0.425 | 真实全库语义检索候选池，对照组 |
| Dify Full-text Top-20 | Global Corpus Retrieval | 500 | 19.952 | 0.849 | 0.710 | 0.988 | 0 | 0.333 | 真实全库关键词 / 全文检索候选池，对照组 |
| Dify Hybrid Top-20 | Global Corpus Retrieval | 500 | 19.906 | 0.922 | 0.846 | 0.998 | 0 | 0.970 | 后续主实验固定候选池，语义 0.7 + 关键词 0.3 |

## 校验结果

- 四个候选池样本数：[500]。
- 样本数是否一致：是。
- 候选解析是否全部成功：是。

## 主候选池选择

后续主实验固定候选池选择：`Dify Hybrid Top-20`。

选择理由：

- Candidate Context 是诊断候选池，gold evidence 天然在候选集合中，不代表真实全库检索难度，因此不作为主实验候选池。
- 在三个真实全库检索候选池中，Dify Hybrid Top-20 的 Evidence Recall 和 All-support Hit 最高，说明其 first-stage retrieval 上限最好。
- Hybrid 延迟最高，但当前任务是 500 条离线复现实验，效率代价可以接受。
- 后续 BGE rerank、LLM rerank、SetR-Selection only、SetR-CoT、SetR-CoT + IRI 都应固定使用同一个 Hybrid Top-20 候选池，以保证后检索方法对比公平。

## 阶段结论

First-stage retrieval 决定后续 reranking / SetR 的理论上限。Candidate Context 用于诊断选择能力；Dify Vector、Full-text、Hybrid 用于比较真实全局检索质量。当前实验中 Hybrid Top-20 是真实检索条件下的最优候选池，后续主实验应固定使用该候选池。
