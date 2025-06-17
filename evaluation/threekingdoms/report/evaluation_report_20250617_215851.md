# ApeRAG Evaluation Report

**Task Name:** Demo Evaluation - Three Kingdoms Q&A (All Metrics)

**Dataset:** ./evaluation/threekingdoms/datasets/qa.csv

**Timestamp:** 20250617_215851

**Total Samples:** 1297

## Bot Configuration

**Bot ID:** bot61b56647225effd1

**Bot Title:** 三国演义

**Bot Type:** knowledge

**Bot Description:** 

## Ragas Evaluation Results

- **Faithfulness**: nan (Range: 0.000 - 1.000)
- **Answer Relevancy**: 0.611 (Range: 0.000 - 1.000)
- **Context Precision**: nan (Range: 0.000 - 1.000)
- **Context Recall**: 0.743 (Range: 0.000 - 1.000)
- **Answer Correctness**: nan (Range: 0.024 - 0.980)


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

{"total_calls": 1297, "calls_with_response_time": 1297, "successful_calls": 1292, "failed_calls": 5, "average_response_time": 10.666572089437167, "min_response_time": 4.135, "max_response_time": 30.225, "total_time": 13834.544000000005, "average_successful_response_time": 10.591356037151707, "min_successful_response_time": 4.135, "max_successful_response_time": 28.55}

## Sample Results

### Sample 1

**Question:** 赵云营寨前的河流叫什么名字？

**Ground Truth:** 汉水。

**Bot Response:** 
```
赵云营寨前的河流叫汉水。
```

**Response Time:** 10.80 seconds

**Evaluation Metrics:**

- Faithfulness: 0.000
- Answer Relevancy: 0.820
- Context Precision: 0.500
- Context Recall: 1.000
- Answer Correctness: 0.912

---

### Sample 2

**Question:** 黄忠劫粮时，曹操派谁领兵将其围困？

**Ground Truth:** 徐晃。

**Bot Response:** 
```
曹操派曹洪领兵将黄忠围困。
```

**Response Time:** 8.90 seconds

**Evaluation Metrics:**

- Faithfulness: 0.000
- Answer Relevancy: 0.690
- Context Precision: 0.500
- Context Recall: 0.000
- Answer Correctness: 0.144

---

### Sample 3

**Question:** 赵云救援黄忠时，杀了曹操哪两位将领？

**Ground Truth:** 慕容烈、焦炳。

**Bot Response:** 
```
赵云救援黄忠时，杀了曹操的将领夏侯渊和夏侯恩。
```

**Response Time:** 9.63 seconds

**Evaluation Metrics:**

- Faithfulness: 0.500
- Answer Relevancy: 0.730
- Context Precision: 0.000
- Context Recall: 0.000
- Answer Correctness: 0.105

---

### Sample 4

**Question:** 曹操在高处看到赵云英勇，说了句什么评价？

**Ground Truth:** “昔日当阳长坂英雄尚在！”

**Bot Response:** 
```
曹操在高处看到赵云英勇，评价道：“真虎将也！吾当生致之。”
```

**Response Time:** 9.40 seconds

**Evaluation Metrics:**

- Faithfulness: 1.000
- Answer Relevancy: 0.699
- Context Precision: 1.000
- Context Recall: 1.000
- Answer Correctness: 0.444

---

### Sample 5

**Question:** 赵云回营后，面对曹军追击，采取了什么策略？

**Ground Truth:** 空营计（偃旗息鼓，大开寨门）。

**Bot Response:** 
```
赵云在面对曹军追击时，采取了以下策略：

1. **单骑突围**：赵云在长坂坡战役中，面对曹军的追击，展现出勇猛和决心，选择单骑突围，以保护刘备的家眷和小主人阿斗。

2. **寻找并保护主母**：赵云首先寻找刘备的妻子和儿子，表现出对主人的忠诚。他在战斗中不断询问百姓，试图找到甘夫人和阿斗的下落。

3. **英勇作战**：在寻找过程中，赵云多次与曹军交战，展示了其卓越的战斗能力。他杀退敌军，保护百姓，并救下了被曹军抓住的糜竺。

4. **突破重围**：最终，赵云成功突破了曹军的重围，带着阿斗逃出险境，确保了刘备的血脉得以保存。

这些策略体现了赵云的勇气、智慧和对刘备的忠诚。
```

**Response Time:** 12.86 seconds

**Evaluation Metrics:**

- Faithfulness: 1.000
- Answer Relevancy: 0.725
- Context Precision: 1.000
- Context Recall: 0.000
- Answer Correctness: 0.074

---

### Sample 6

**Question:** 曹操任命谁为副先锋，协助徐晃渡汉水作战？

**Ground Truth:** 王平。

**Bot Response:** 
```
根据提供的信息，曹操任命朱赞为副先锋，协助徐晃渡汉水作战。
```

**Response Time:** 7.10 seconds

**Evaluation Metrics:**

- Faithfulness: 0.000
- Answer Relevancy: 0.666
- Context Precision: 1.000
- Context Recall: 0.000
- Answer Correctness: 0.107

---

### Sample 7

**Question:** 徐晃欲效仿哪位古人采用背水列阵？

**Ground Truth:** 韩信。

**Bot Response:** 
```
根据提供的信息，徐晃想要效仿的古人是韩信，采用背水列阵的策略。
```

**Response Time:** 10.16 seconds

**Evaluation Metrics:**

- Faithfulness: 0.000
- Answer Relevancy: 0.889
- Context Precision: 0.000
- Context Recall: 0.000
- Answer Correctness: 0.745

---

### Sample 8

**Question:** 黄忠和赵云用什么策略击败了渡过汉水的徐晃？

**Ground Truth:** 待其日暮兵疲，分兵两路夹攻。

**Bot Response:** 
```
黄忠和赵云采用了"待日暮兵疲"的策略来击败渡过汉水的徐晃。他们在战斗中选择不立即与徐晃交战，而是耐心等待，直到徐晃的士兵疲惫不堪。在徐晃的部队疲惫并且出现退动的时机，黄忠和赵云分别从左右夹攻，最终成功地将徐晃大败，导致其军士被逼入汉水，死者无数。
```

**Response Time:** 8.55 seconds

**Evaluation Metrics:**

- Faithfulness: 1.000
- Answer Relevancy: 0.927
- Context Precision: 1.000
- Context Recall: 1.000
- Answer Correctness: 0.732

---

### Sample 9

**Question:** 王平因何事投降了刘备？

**Ground Truth:** 徐晃欲杀他，他夜晚放火烧营后投奔赵云。

**Bot Response:** 
```
王平因在与曹操的战斗中，徐晃不听他的劝告，导致大败而投降刘备。王平在战斗中表现出色，但由于徐晃的鲁莽和错误决策，使得局势对蜀汉不利，最终王平选择投降刘备以寻求更好的归属。
```

**Response Time:** 9.28 seconds

**Evaluation Metrics:**

- Faithfulness: 0.600
- Answer Relevancy: 0.910
- Context Precision: 0.241
- Context Recall: 0.000
- Answer Correctness: 0.147

---

### Sample 10

**Question:** 诸葛亮如何利用疑兵之计骚扰曹操，使其退兵？

**Ground Truth:** 夜间令赵云在土山后鸣鼓角，让曹兵惊疑不安。

**Bot Response:** 
```

```

**Response Time:** 30.05 seconds

**Error:** API call failed: 

---

*(1287 more samples not shown)*

