# SETR Reproduction

本仓库是NLP课程期末作业的论文复现实验支撑材料，复现对象为 ACL 2025 论文 **Shifting from Ranking to Set Selection for Retrieval Augmented Generation**（SETR）。

本复现不进行官方模型微调，主要在HotpotQA dev distractor的500个问题上，用prompt-level方式复现和分析SETR的核心思想：将RAG后检索阶段从传统reranking转为证据集合选择（set selection）。

## 目录结构

```text
.
├── BGE-Reranker/
│   ├── run_bge_reranker_top5.py
│   └── requirements.txt
├── paper/
│   ├── SETR_2025_ACL.pdf
│   └── SETR_2025_ACL.txt
└── setr_reproduction/
    ├── configs/
    ├── prompts/
    ├── scripts/
    ├── requirements.txt
    └── results/
```

## 包含内容

- `paper/`：SETR原论文PDF和文本版。
- `setr_reproduction/scripts/`：实验流程脚本。
- `setr_reproduction/prompts/`：LLM Listwise、SetR和答案生成提示词。
- `setr_reproduction/results/`：候选池、重排序/集合选择结果、答案生成结果、评估表格、图表和案例分析。
- `BGE-Reranker/`：本地BGE-Reranker Top-5基线实现。

## 不包含内容

为控制仓库体积，本仓库不包含：

- HotpotQA完整原始数据；
- 全量去重知识库语料；
- Dify知识库索引；
- `.env`、API Key或本地服务配置；
- 作业报告、实施方案和过程记录。

## 主要实验方法

主实验比较以下方法：

1. Retrieval Top-5
2. BGE-Reranker Top-5
3. LLM Listwise Top-5
4. SetR-Selection only
5. SetR-CoT
6. SetR-CoT + IRI

主候选池使用Dify Hybrid Top-20，最终答案统一由Qwen3-14B生成。

## 结果位置

主要结果文件位于：

```text
setr_reproduction/results/tables/
```

其中：

- `main_results.csv`：主实验结果。
- `evidence_results.csv`：证据覆盖结果。
- `efficiency_results.csv`：效率分析结果。
- `setr_ablation.csv`：SetR消融结果。
- `report_tables.md`：报告用表格汇总。

图表位于：

```text
setr_reproduction/results/figures/
```

案例分析位于：

```text
setr_reproduction/results/cases/
```

## 复核已有结果

如果只复核已有实验结果，可运行：

```bash
python setr_reproduction/scripts/13_evaluate_results.py --force
python setr_reproduction/scripts/14_setr_ablation_analysis.py --force
python setr_reproduction/scripts/15_efficiency_analysis.py --force
python setr_reproduction/scripts/16_case_analysis.py --force
python setr_reproduction/scripts/17_generate_report_tables.py --force
```

重新运行涉及Dify检索、Qwen3-14B调用或BGE-Reranker推理的脚本时，需要自行准备对应环境和服务配置。

