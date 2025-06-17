# ApeRAG Evaluation Report

**Task Name:** Demo Evaluation - Three Kingdoms Q&A (All Metrics)

**Dataset:** ./evaluation/threekingdoms/datasets/qa.csv

**Timestamp:** 20250617_191922

**Total Samples:** 100

## Bot Configuration

**Bot ID:** bot61b56647225effd1

**Bot Title:** 三国演义

**Bot Type:** knowledge

**Bot Description:** 

## Ragas Evaluation Results

- **Faithfulness**: 0.649 (Range: 0.000 - 1.000)
- **Answer Relevancy**: 0.524 (Range: 0.000 - 1.000)
- **Context Precision**: 0.790 (Range: 0.000 - 1.000)
- **Context Recall**: 0.585 (Range: 0.000 - 1.000)
- **Answer Correctness**: 0.375 (Range: 0.069 - 0.935)


## Ragas Evaluation Metrics Explanation

Ragas (Retrieval Augmented Generation Assessment) is a framework specifically designed for evaluating RAG systems, providing the following core metrics:

### 1. Faithfulness
- **Definition**: Measures the consistency between generated answers and retrieved context information
- **Calculation**: Analyzes whether statements in the answer can find support in the retrieved context
- **Score Range**: 0-1, higher scores indicate answers are more faithful to source materials
- **Significance**: Ensures AI doesn't generate hallucinated content that contradicts facts

### 2. Answer Relevancy  
- **Definition**: Evaluates how relevant the generated answer is to the user's question
- **Calculation**: Analyzes whether the answer directly addresses the user's specific question
- **Score Range**: 0-1, higher scores indicate more relevant answers
- **Significance**: Avoids irrelevant responses and ensures targeted answers

### 3. Context Precision
- **Definition**: Measures the proportion of relevant content in retrieved context information
- **Calculation**: Evaluates the ratio of useful information vs irrelevant information in retrieval results
- **Score Range**: 0-1, higher scores indicate more precise retrieval
- **Significance**: Optimizes retrieval strategy and reduces noise

### 4. Context Recall
- **Definition**: Evaluates whether the retrieval system can find all relevant information needed to answer the question
- **Calculation**: Checks if all information the answer depends on can be found in the retrieved context
- **Score Range**: 0-1, higher scores indicate more complete retrieval coverage
- **Significance**: Ensures important information is not missed

### 5. Answer Correctness
- **Definition**: Comprehensively evaluates answer correctness, combining factual accuracy and semantic similarity
- **Calculation**: Compares generated answers with standard answers at semantic and factual levels
- **Score Range**: 0-1, higher scores indicate more accurate answers
- **Significance**: Comprehensively measures RAG system output quality
## Performance Statistics

**Response Time Statistics:**

{"total_calls": 100, "calls_with_response_time": 100, "successful_calls": 100, "failed_calls": 0, "average_response_time": 9.378760000000002, "min_response_time": 5.002, "max_response_time": 18.087, "total_time": 937.8760000000001, "average_successful_response_time": 9.378760000000002, "min_successful_response_time": 5.002, "max_successful_response_time": 18.087}

## Sample Results

### Sample 1

**Question:** 赵云营寨前的河流叫什么名字？

**Ground Truth:** 汉水。

**Bot Response:** 
```
赵云营寨前的河流叫汉水。
```

**Response Time:** 9.47 seconds

**Evaluation Metrics:**

- Faithfulness: 0.000
- Answer Relevancy: 0.585
- Context Precision: 1.000
- Context Recall: 1.000
- Answer Correctness: 0.920

---

### Sample 2

**Question:** 黄忠劫粮时，曹操派谁领兵将其围困？

**Ground Truth:** 徐晃。

**Bot Response:** 
```
曹操派夏侯尚和韩浩引兵来围困黄忠。
```

**Response Time:** 10.56 seconds

**Evaluation Metrics:**

- Faithfulness: 0.500
- Answer Relevancy: 0.514
- Context Precision: 0.000
- Context Recall: 0.000
- Answer Correctness: 0.096

---

### Sample 3

**Question:** 赵云救援黄忠时，杀了曹操哪两位将领？

**Ground Truth:** 慕容烈、焦炳。

**Bot Response:** 
```
赵云在救援黄忠时，杀了曹操的两位将领是夏侯尚和韩浩。
```

**Response Time:** 10.42 seconds

**Evaluation Metrics:**

- Faithfulness: 0.000
- Answer Relevancy: 0.497
- Context Precision: 0.000
- Context Recall: 0.000
- Answer Correctness: 0.102

---

### Sample 4

**Question:** 曹操在高处看到赵云英勇，说了句什么评价？

**Ground Truth:** “昔日当阳长坂英雄尚在！”

**Bot Response:** 
```
曹操在高处看到赵云英勇，评价道：“我向曾闻云长言：翼德于百万军中，取上将之首，如探囊取物。”这句话表明了曹操对张飞的勇猛和能力的高度认可。
```

**Response Time:** 8.69 seconds

**Evaluation Metrics:**

- Faithfulness: 0.667
- Answer Relevancy: 0.515
- Context Precision: 1.000
- Context Recall: 1.000
- Answer Correctness: 0.514

---

### Sample 5

**Question:** 赵云回营后，面对曹军追击，采取了什么策略？

**Ground Truth:** 空营计（偃旗息鼓，大开寨门）。

**Bot Response:** 
```
赵云在面对曹军的追击时，采取了以下策略：

1. **埋伏与追击**：赵云奉命在乌林小路埋伏，准备伏击曹操，以此来对抗曹军的追击。

2. **勇敢冲击**：在长坂坡战役中，赵云单骑救主，勇敢地冲入敌阵，保护了刘备的家眷，并成功突围，展现了他的忠诚和战斗力。

3. **保护行动**：赵云在关键时刻选择保护刘备的儿子阿斗，显示出他对刘备及其家族的极高忠诚。

这些策略体现了赵云的军事才能和对蜀汉的忠诚。
```

**Response Time:** 10.35 seconds

**Evaluation Metrics:**

- Faithfulness: 0.800
- Answer Relevancy: 0.569
- Context Precision: 1.000
- Context Recall: 0.000
- Answer Correctness: 0.115

---

### Sample 6

**Question:** 曹操任命谁为副先锋，协助徐晃渡汉水作战？

**Ground Truth:** 王平。

**Bot Response:** 
```
根据提供的信息，曹操并没有明确提到任命谁为副先锋协助徐晃渡汉水作战的信息。因此，无法回答这个问题。
```

**Response Time:** 7.05 seconds

**Evaluation Metrics:**

- Faithfulness: 1.000
- Answer Relevancy: 0.000
- Context Precision: 1.000
- Context Recall: 1.000
- Answer Correctness: 0.077

---

### Sample 7

**Question:** 徐晃欲效仿哪位古人采用背水列阵？

**Ground Truth:** 韩信。

**Bot Response:** 
```
根据提供的信息，无法确定徐晃是否想效仿哪位古人采用背水列阵的具体情况。因此，无法回答此问题。
```

**Response Time:** 6.40 seconds

**Evaluation Metrics:**

- Faithfulness: 1.000
- Answer Relevancy: 0.000
- Context Precision: 0.000
- Context Recall: 0.000
- Answer Correctness: 0.110

---

### Sample 8

**Question:** 黄忠和赵云用什么策略击败了渡过汉水的徐晃？

**Ground Truth:** 待其日暮兵疲，分兵两路夹攻。

**Bot Response:** 
```
黄忠和赵云采用了分兵夹击的策略，成功击败了渡过汉水的曹军将领徐晃。具体操作是黄忠和赵云分别引兵从两侧夹击徐晃，趁着徐晃军队在长时间的搦战后疲惫不堪时，发动攻击，最终导致徐晃大败，军士被逼入汉水，死者无数。
```

**Response Time:** 11.83 seconds

**Evaluation Metrics:**

- Faithfulness: 1.000
- Answer Relevancy: 0.504
- Context Precision: 1.000
- Context Recall: 1.000
- Answer Correctness: 0.802

---

### Sample 9

**Question:** 王平因何事投降了刘备？

**Ground Truth:** 徐晃欲杀他，他夜晚放火烧营后投奔赵云。

**Bot Response:** 
```
王平投降刘备的原因是因为他在与曹操的战争中感受到曹操的强大压力和威胁。在《三国演义》中，王平原本是曹操的部下，但在与刘备的关系中，王平最终选择投降刘备，成为蜀汉的一员。王平在军事上展现了卓越的才能和忠诚，熟悉汉水地区的地理，参与了多次与魏国的战斗，成为蜀汉的中坚力量。

具体来说，王平因曹操的强势和对蜀汉的威胁，以及他与刘备之间的交情和对刘备理想的认同，使他最终选择了投降刘备。此外，王平的投降也与刘备在当时的政治形势和军事环境有关，刘备在一定程度上代表了对抗曹操的重要力量，使得王平在权衡利弊后决定投降刘备。
```

**Response Time:** 13.59 seconds

**Evaluation Metrics:**

- Faithfulness: 0.583
- Answer Relevancy: 0.632
- Context Precision: 1.000
- Context Recall: 0.000
- Answer Correctness: 0.116

---

### Sample 10

**Question:** 诸葛亮如何利用疑兵之计骚扰曹操，使其退兵？

**Ground Truth:** 夜间令赵云在土山后鸣鼓角，让曹兵惊疑不安。

**Bot Response:** 
```
诸葛亮利用疑兵之计骚扰曹操的策略主要体现在《三国演义》第四十六回中。具体过程如下：

1. **制造假象**：诸葛亮在得知曹操的军队即将来袭时，故意在自己的阵地上制造出一种繁忙的气氛，以此让曹操误以为蜀军正在全力备战。

2. **利用天气条件**：诸葛亮提前预测到会有大雾天气，因此选择在大雾弥漫的夜晚出动，利用视线不清的条件来掩护自己的行动。

3. **借船取箭**：他设计了一个巧妙的计划，借用鲁肃的船只，命令士兵们在船上悬挂草帘，模拟出一支庞大的船队。然后通过大雾悄然接近曹操的水寨，趁机擂鼓呐喊，制造出一种蜀军大规模出击的假象。

4. **引诱曹操出兵**：曹操在听到擂鼓声后，因不明情况而心生疑虑，命令手下发箭攻击。实际情况是，诸葛亮利用这次机会成功地用船只收集到大量箭矢，而曹操则因惧怕埋伏而未敢轻举妄动。

5. **让曹操自乱阵脚**：最终，曹操被诸葛亮的计策迷惑，导致其军队的士气受挫，未能及时展开有效的反击，最终选择退兵。

通过这些巧妙的计划和利用气候、心理战等手段，诸葛亮成功地骚扰了曹操，使其不能如愿进攻，反而造成了曹军的混乱和退却。
```

**Response Time:** 15.40 seconds

**Evaluation Metrics:**

- Faithfulness: 1.000
- Answer Relevancy: 0.503
- Context Precision: 1.000
- Context Recall: 1.000
- Answer Correctness: 0.129

---

*(90 more samples not shown)*

