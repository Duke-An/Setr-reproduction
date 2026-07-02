# 环节 13：案例分析

本文件从 500 条 HotpotQA dev distractor 样本中选取 5 个典型案例，用于解释定量结果背后的原因。

案例覆盖：SetR 成功、reranker 成功而 SetR 失败、候选池缺失、SetR 少选但答对、以及格式/冗余问题。

## Case 1: SetR-CoT + IRI 成功，Top-5 / reranker 失败

- qid: `5a78ae8b5542990784727730`
- sample_index: 51
- Question: How far from Sacramento is the flight school in Atwater?
- Gold answer: about 115 miles (185 km)
- Gold titles: Sierra Academy of Aeronautics, Castle Air Force Base
- Top-20 all-support hit: 1
- Top-20 missing gold titles: []

| Method | Generated answer | EM | F1 | All-support | Selected count | Selected titles | Missing gold titles |
|---|---|---:|---:|---:|---:|---|---|
| Retrieval Top-5 | Insufficient Information | 0.0 | 0.0000 | 1.0 | 5 | Sierra Academy of Aeronautics, ATP Flight School, Nevada Union High School, Castle Air Force Base, Pacific Flying Club | [] |
| BGE-Reranker Top-5 | Insufficient Information | 0.0 | 0.0000 | 1.0 | 5 | Sierra Academy of Aeronautics, Castle Air Force Base, ATP Flight School, Nevada Union High School, Camp Far West Reservoir | [] |
| LLM Listwise Top-5 | Insufficient Information | 0.0 | 0.0000 | 1.0 | 5 | Sierra Academy of Aeronautics, Castle Air Force Base, Nevada Union High School, Northern California TRACON, ATP Flight School | [] |
| SetR-Selection only | about 115 miles (185 km) | 1.0 | 1.0000 | 1.0 | 2 | Sierra Academy of Aeronautics, Castle Air Force Base | [] |
| SetR-CoT | about 115 miles (185 km) | 1.0 | 1.0000 | 1.0 | 2 | Sierra Academy of Aeronautics, Castle Air Force Base | [] |
| SetR-CoT + IRI | about 115 miles (185 km) | 1.0 | 1.0000 | 1.0 | 2 | Sierra Academy of Aeronautics, Castle Air Force Base | [] |

分析：

SetR-CoT + IRI 给出完全匹配答案，而至少主要 Top-5 baseline 未能给出 EM 正确答案；用于展示集合选择在部分样本上能压缩上下文并保留关键证据。

## Case 2: reranker 成功，SetR 失败

- qid: `5a727e1b5542991f9a20c497`
- sample_index: 9
- Question: In which song was written by singer-songwriter Taylor Swift and shares the optimistic lyrical message to a song called "Yodel It!"?
- Gold answer: Shake It Off
- Gold titles: Yodel It!, Shake It Off
- Top-20 all-support hit: 1
- Top-20 missing gold titles: []

| Method | Generated answer | EM | F1 | All-support | Selected count | Selected titles | Missing gold titles |
|---|---|---:|---:|---:|---:|---|---|
| Retrieval Top-5 | Insufficient Information | 0.0 | 0.0000 | 1.0 | 5 | Yodel It!, You Belong with Me, Shake It Off, Taylor Swift, Our Song (Taylor Swift song) | [] |
| BGE-Reranker Top-5 | Insufficient Information | 0.0 | 0.0000 | 0.0 | 5 | Yodel It!, Taylor Swift, Taylor Swift (album), Our Song (Taylor Swift song), Change (Taylor Swift song) | Shake It Off |
| LLM Listwise Top-5 | Shake It Off | 1.0 | 1.0000 | 1.0 | 5 | Yodel It!, Shake It Off, You Belong with Me, Our Song (Taylor Swift song), Fearless (Taylor Swift song) | [] |
| SetR-Selection only | Insufficient Information | 0.0 | 0.0000 | 1.0 | 2 | Yodel It!, Shake It Off | [] |
| SetR-CoT | Insufficient Information | 0.0 | 0.0000 | 1.0 | 2 | Yodel It!, Shake It Off | [] |
| SetR-CoT + IRI | Insufficient Information | 0.0 | 0.0000 | 1.0 | 2 | Yodel It!, Shake It Off | [] |

分析：

BGE 或 LLM Listwise Top-5 能答对，但三个 SetR 变体均未 EM 命中；用于说明过度压缩上下文或漏选证据会损伤答案生成。

## Case 3: 三者都失败，原因是 top-20 候选池没召回完整 gold evidence

- qid: `5a7344e95542991f9a20c6ce`
- sample_index: 17
- Question: What song was number 4 on the charts when a song from FutureSex/LoveSounds was number 1?
- Gold answer: Rudebox
- Gold titles: Rudebox (song), SexyBack
- Top-20 all-support hit: 0
- Top-20 missing gold titles: Rudebox (song)

| Method | Generated answer | EM | F1 | All-support | Selected count | Selected titles | Missing gold titles |
|---|---|---:|---:|---:|---:|---|---|
| Retrieval Top-5 | Insufficient Information | 0.0 | 0.0000 | 0.0 | 5 | FutureSex/LoveSound, FutureSex/LoveSounds, FutureSex/LoveShow, All 4 Love, Love in the Future | Rudebox (song), SexyBack |
| BGE-Reranker Top-5 | Insufficient Information | 0.0 | 0.0000 | 0.0 | 5 | FutureSex/LoveSound, FutureSex/LoveSounds, SexyBack, FutureSex/LoveShow, Love in the Future | Rudebox (song) |
| LLM Listwise Top-5 | Insufficient Information | 0.0 | 0.0000 | 0.0 | 5 | All 4 Love, Love in the Future, FutureSex/LoveSound, SexyBack, FutureSex/LoveSounds | Rudebox (song) |
| SetR-Selection only | Insufficient Information | 0.0 | 0.0000 | 0.0 | 2 | All 4 Love, Love in the Future | Rudebox (song), SexyBack |
| SetR-CoT | Insufficient Information | 0.0 | 0.0000 | 0.0 | 2 | Love in the Future, Interstate Love Song | Rudebox (song), SexyBack |
| SetR-CoT + IRI | Insufficient Information | 0.0 | 0.0000 | 0.0 | 2 | Love in the Future, SexyBack | Rudebox (song) |

分析：

所有方法均未 EM 命中，并且 Hybrid Top-20 本身没有覆盖全部 gold titles；用于说明 first-stage retrieval 上限会限制后续 rerank / SetR。

## Case 4: SetR 选更少 passage 但答案正确

- qid: `5a71231a5542994082a3e5c4`
- sample_index: 0
- Question: What edible, juicy fruit is grown on a deciduous tree called 'pesco' in Italian?
- Gold answer: peach
- Gold titles: Pesco, Peach
- Top-20 all-support hit: 1
- Top-20 missing gold titles: []

| Method | Generated answer | EM | F1 | All-support | Selected count | Selected titles | Missing gold titles |
|---|---|---:|---:|---:|---:|---|---|
| Retrieval Top-5 | Peach | 1.0 | 1.0000 | 1.0 | 5 | Peach, Pesco, Quince, Tomato, Apple | [] |
| BGE-Reranker Top-5 | Peach | 1.0 | 1.0000 | 1.0 | 5 | Peach, Pesco, Amelanchier interior, Apple, Tomato | [] |
| LLM Listwise Top-5 | Peach | 1.0 | 1.0000 | 1.0 | 5 | Pesco, Peach, Quince, Tomato, Apple | [] |
| SetR-Selection only | Peach | 1.0 | 1.0000 | 1.0 | 2 | Peach, Pesco | [] |
| SetR-CoT | Peach | 1.0 | 1.0000 | 1.0 | 2 | Peach, Pesco | [] |
| SetR-CoT + IRI | Peach | 1.0 | 1.0000 | 1.0 | 3 | Peach, Pesco, Mote con huesillo | [] |

分析：

SetR-Selection only 用少于 5 个 passage 覆盖完整 gold evidence 并答对；用于展示 SetR 的上下文压缩能力。

## Case 5: Qwen3-14B 输出格式错误或选择冗余 passage

- qid: `5a77b0395542997042120ae1`
- sample_index: 48
- Question: Which star in the movie Hush was born April 20, 1949?
- Gold answer: Jessica Lange
- Gold titles: Hush (1998 film), Jessica Lange
- Top-20 all-support hit: 0
- Top-20 missing gold titles: Jessica Lange

| Method | Generated answer | EM | F1 | All-support | Selected count | Selected titles | Missing gold titles |
|---|---|---:|---:|---:|---:|---|---|
| Retrieval Top-5 | Insufficient Information | 0.0 | 0.0000 | 0.0 | 5 | Hush (1998 film), Hush… Hush, Sweet Charlotte, Walter Huston, Diana Quick, Marian Mercer | Jessica Lange |
| BGE-Reranker Top-5 | Insufficient Information | 0.0 | 0.0000 | 0.0 | 5 | Hush (1998 film), Hush… Hush, Sweet Charlotte, Walter Huston, Sandra Dee, Humphrey Bogart | Jessica Lange |
| LLM Listwise Top-5 | Insufficient Information | 0.0 | 0.0000 | 0.0 | 5 | Hush (1998 film), Hush… Hush, Sweet Charlotte, Walter Huston, Diana Quick, Marian Mercer | Jessica Lange |
| SetR-Selection only | Insufficient Information | 0.0 | 0.0000 | 0.0 | 5 | Hush (1998 film), Hush… Hush, Sweet Charlotte, Walter Huston, Diana Quick, Marian Mercer | Jessica Lange |
| SetR-CoT | Insufficient Information | 0.0 | 0.0000 | 0.0 | 5 | Hush (1998 film), Walter Huston, Lauren Bacall, Pat Hingle, Sandra Dee | Jessica Lange |
| SetR-CoT + IRI | Insufficient Information | 0.0 | 0.0000 | 0.0 | 1 | Hush (1998 film) | Jessica Lange |

分析：

该样本触发 SetR 解析失败 / fallback，用于展示 prompt 输出格式问题。

## 总结

这些案例说明：SetR 的主要优势是能够用更少 passage 保留回答所需信息；但当 first-stage retrieval 没有召回完整 gold evidence，或 prompt-level SetR 过度压缩上下文时，后续生成仍会失败。IRI 在部分样本上能帮助模型围绕信息需求选择证据，但在当前未微调设置下也可能引入冗余或漏选。