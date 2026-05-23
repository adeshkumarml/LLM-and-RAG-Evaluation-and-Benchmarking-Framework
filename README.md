# LLM and RAG Evaluation Benchmarking Framework - Development Repository
This repository contains the experimentation and development work for this framework. It includes evaluation and orchestration pipelines, experimentation notebooks, workflow details, engineering decisions etc. For the deployment repository and deployment notes visit: https://github.com/adeshkumarml/LLM-and-RAG-Evaluation-and-Benchmarking-Framework-Deployment-Repo

## 1. Problem Statement and Motivation
Retrieval-Augmented Generation (RAG) systems are widely used to improve Large Language Model (LLM) responses by anchoring generations with external knowledge. The primary focus of this project is not to make another RAG application, but to see the metrics and systemetically evaluate whether retrieval actually improves generation quality.<br>
From RAG-based generations, though it seems like “the answer looks better,” quantitatively measuring retrieval effectiveness, semantic alignment, or response quality are what actually proves if it is so in reality, which in turn helps us better to understand the downstream performance as well.<br>
The motivation behind this project was to design an end-to-end evaluation and benchmarking framework capable of comparing baseline LLM responses against retrieval-enhanced responses using multiple evaluation strategies. Instead of focusing solely on chatbot interaction, the project emphasizes measurable evaluation, modular experimentation, and analysis of retrieval behavior under both successful and failure-case scenarios.<br>
The framework combines:<br>
- Retrieval Evaluation Metrics,
- Semantic Similarity Scoring,
- Exact Match Benchmarking, and
- LLM-as-Judge Qualitative Evaluation<br>
aimed at providing a more comprehensive view of RAG system performance.<br>
The project was also developed with deployment and usability in mind, resulting in both:
- a local evaluation pipeline for experimentation
- a deployed cloud-based interface for interactive benchmarking (link above)

## 2. Dataset
This project uses a subset of the Stanford Question Answering Dataset (SQuAD) v2 dataset for benchmarking retrieval and generation quality in question-answering (QA) tasks.<br>
SQuAD v2 is a widely used reading comprehension and QA benchmark consisting of:
- context passages,
- natural language questions,
- human-annotated ground truth answers<br>
In addition, unlike SQuAD v1, the v2 version additionally includes unanswerable questions, making it more suitable for evaluating hallucination resistance and grounded generation behavior in retrieval-based systems.<br>
More details here: https://rajpurkar.github.io/SQuAD-explorer/<br>
The dataset was chosen because:
- it is a standardized and well-established QA dataset
- contains realistic context-question-answer structures
- suitable for evaluating both retrieval and generation quality
- allows controlled comparison between baseline and RAG-enhanced responses<br>
For experimentation and efficiency, a smaller curated subset of 150 samples was used during evaluation.
### 2.1 Dataset Access
Please check these links:
- downloaded directly from the official SQuAD website: https://rajpurkar.github.io/SQuAD-explorer/
- loaded programmatically using the Hugging Face Datasets Library: https://huggingface.co/datasets/rajpurkar/squad_v2

```python
from datasets import load_dataset
dataset = load_dataset("squad_v2")
```
### 2.2 Dataset Schema
Each sample contains:
- an UID
- a context
- a question
- one or more acceptable ground-truth answers

## 3. System Architecture and Overview of Pipelines
This framework is designed as a modular NLP evaluation system separating retrieval, generation, evaluation, orchestration, and deployment responsibilities into independent components. The architecture supports both local experimentation workflows and cloud-based online inference through a deployed Streamlit and FastAPI stack.<br>
![Architecture Diagram](assets/RAG%20Eval%20Framework%20Arch%20Diag.png)<br>
### 3.1 Evaluation Workflow
The evaluation workflow follows the sequence:<br>
Dataset Sample ➡️ Context Chunking ➡️ Embedding Generation ➡️ Top-K Semantic Retrieval ➡️ Baseline LLM Generation ➡️ RAG-Enhanced Generation ➡️ Metric Evaluation ➡️ LLM-as-Judge Scoring ➡️ Aggregation & Reporting<br>
For each evaluation sample, the framework does the following:<br>
  (i) Generates a baseline response without retrieval augmentation<br>
 (ii) Retrieves semantically relevant chunks from the corpus<br>
(iii) Generates a retrieval-grounded response<br>
 (iv) Evaluates both outputs across multiple metrics<br>
  (v) Aggregates evaluation results into structured reports
### 3.2 Local and Online Workflows
This framework evaluation has been made to be utilized in either offline mode or online mode.<br>
**Local Workflow:** The local workflow is intended for experimentation and large-scale evaluation runs. This can be run directly using the *main.py* driver module.<br>
*main.py* ➡️ Evaluation Orchestrator ➡️ Aggregate Metrics ➡️ JSON Result Generation 
**Online Workflow:** The deployed workflow supports interactive benchmarking through the web interface.<br>
Uploads Corpus and QA Dataset ➡️ Streamlit Frontend ➡️ FastAPI Backend ➡️ Evaluation Pipeline Execution ➡️ Results Returned to UI
### 3.3 Modules and their Functions
|**Package**|**Modules**|**Function**
|---|---|
|*retrieval*|`chunking.py`, `embedding.py`, `retirever.py`|Chunking corpus/context, embedding generation and semantic retrieval of top-k chunks
|*pipelines*|`baseline.py`, `rag.py`|Baseline generation, i.e, answer generated by LLM itself and RAG-grounded answer generation
|*evaluation*|`metrics.py`, `retrieval_metrics.py`, `parser.py`, `llm_judge.py`|Compute metrics for the generated answer, metrics for retireval and LLM-as-Judge evaluation. The parser module here is to parse the llm-judge response and extract the scores and structure it for further utilization.
|*orchestration*|`evaluator.py`|Co-ordinates all the module functions and end-to-end evaluation
|*utils.py*|`utils.py`, `openai_client.py`|Logging, configuration and api key loading
|*app*|`fast_app.py`, `streamlit_ui.py`|Streamlit frontend and FastAPI backend
### 3.4 Dataflow
The system processes data through the following stages:<br>
- Raw corpus text is chunked into smaller retrieval units. (Note that offline evaluation through `main.py`, the "contexts" are chunked.)
- Sentence-transformer embeddings are generated for all chunks.
- Incoming questions are embedded and matched against the corpus using semantic similarity.
- The top-k retrieved chunks are appended to the generation prompt.
- Baseline and retrieval-augmented responses are generated independently.
- Both responses are evaluated using retrieval, semantic, lexical, and judge-based metrics.
- Evaluation outputs are aggregated into structured JSON reports and visualized through the frontend dashboard.

## 4. Evaluation Details and Choice of Metrics
As it stands out, evaluating retrieval-augmented systems using a single metric often provides an incomplete or a partial picture of the performance. For instance, a response may be semantically correct but can fail lexical matching, or cases where retrieval may succeed while generation still produces weak answers. To address this, this framework combines multiple complementary evaluation strategies as discussed below.<br>
Simply saying, the quantification of the performance is done in three parts: (i) Metrics for the LLM vs RAG (EM and Semnantic Similarity), (ii) Metrics for retrieval quality itself (Precision and Recall at k and MRR) and (iii) Metrics by LLM-as-judge<br>  
### 4.1 Exact Match (EM)
Exact Match evaluates whether the generated response exactly matches any of the ground-truth answers.<br>
It (i) measures strict lexical correctness, (ii) useful for objective QA tasks and (iii) widely used in benchmark datasets such as SQuAD.<br>
However there is quite a big limitation to this.<br>
Exact Match is intentionally strict and may under-score semantically correct responses that differ in wording or phrasing.
**Example:**<br>
Ground Truth: *action*, *work*<br>
Generated: *action or deed*<br>
Although semantically correct, Exact Match assigns a score of 0.
### 4.2 Semantic Similarity
To complement lexical matching, the framework computes embedding-based semantic similarity between generated responses and ground-truth answers. It (i) captures semantic alignment beyond exact wording, (ii) recognizes paraphrases and meaning-preserving responses and (iii) provides softer evaluation than Exact Match<br>
Semantic similarity helps reduce false negatives introduced by strict lexical comparison.
### 4.3 Retrieval Metrics
Retrieval quality is evaluated independently using:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
These metrics rely on lexical overlap between retrieved chunks and ground-truth answers and are aimed at (i) measuring retrieval effectiveness, (ii) evaluating whether relevant information is surfaced and (iii) separates retrieval performance from generation performance.
This distinction is important because strong retrieval does not always guarantee strong generation, and weak retrieval may still occasionally produce acceptable outputs.<br>
However, lexical-overlap retrieval metrics may under-report semantically relevant chunks when wording differs from the reference answer. But here, since the ground-truth answers are directly the copied phrasing from the context itself, lexical-overlap is very likely to succeed in most cases.
### 4.4 LLM-as-Judge Evaluation
Despite having intrinsic/rule-based metrics, evaluation through numerical metrics alone do not fully capture response quality. The idea is to use a proxy-judge to evaluate a natural language in human ways. So to address this, the framework additionally uses LLM-as-Judge evaluation across three qualitative dimensions:<br>
- Correctness
- Completeness
- Fluency<br>
The purpose is to evaluate factual quality and response usefulness by capturing qualitative aspects missed by rigid lexical metrics and thereby
providing interpretable scoring beyond just token overlap.<br>
This approach complements traditional evaluation metrics by allowing assessment of groundedness and response quality in a more human-aligned manner. 
### 4.5 Choosing Multiple Metrics
As it turns out no single metric can adequately evaluate RAG systems. <br>
The framework therefore combines:
- lexical evaluation (Exact Match)
- semantic evaluation (Similarity Scoring)
- retrieval evaluation (Precision@K, Recall@K, MRR)
- qualitative evaluation (LLM-as-Judge)<br>
to provide a more comprehensive and balanced assessment of retrieval and generation performance, with all the metrics complemnenting each other.

## 5. Engineering Decisions & Design Tradeoffs
This framework is designed with an emphasis on modularity, deployment feasibility, and interpretable evaluation. Several implementation choices were intentionally made to balance retrieval quality, inference cost, engineering simplicity, and deployment constraints.
### 5.1 Modular Architecture Design
The system was refactored into independent modules separating:<br>
- retrieval
- generation
- evaluation
- orchestration
- deployment layers
This separation improves maintainability, debugging, component reusability and deployment readiness.<br>
The modular design also allows retrieval and evaluation pipelines to evolve independently without tightly coupling system components.
### 5.2 Embedding Model Choice
This framework uses **all-MiniLM-L6-v2** for embeddings and semantic retrieval.<br>
This model was selected because it provides a rather strong semantic retrieval quality, fast embedding generation, low memory requirements and practical deployment performance. This made it suitable for both local experimentation and free-tier cloud deployment.
**Tradeoffs:** Although efficient, the model is not state-of-the-art for retrieval and may struggle with longer contextual dependencies or very domain-specific retrieval tasks. Therefore this becomes a latency-quality tradeoff rather than maximum retrieval accuracy.
### 5.3 Generation Model Choice
This framework uses **gpt-4o-mini** for both answer generation and LLM-as-Judge evaluation. This model was selected because being a rather new LLM, it has reliable instruction following, comparatively low inference cost than flagship models, fast response generation and stable API-based deployment.<br>
Using a single model for generation and qualitative evaluation simplified orchestration and reduced operational complexity.
**Tradeoffs:** It is worth noting that evaluation may inherit model-specific biases, larger frontier models may provide stronger reasoning quality and API usage introduces external dependency and cost considerations.<br>
The model choice therefore prioritizes reproducible benchmarking under realistic deployment constraints.
### 5.4 Prompt Design for LLM
Prompt construction was intentionally kept constrained and task-oriented. Rather than using complex prompting strategies, the framework used concise and tightly scoped instructions aligned with the retrieval context. This design encouraged the model to rely on retrieved evidence and abstain when sufficient supporting information was unavailable, thus keeping hallucinations in check as much as possible.
### 5.5 Retrieval Design Decisions
#### Chunk Size
Corpus documents/contexts are split into fixed-length chunks before embedding generation.<br> 
Chunking introduces a tradeoff between smaller chunks which gives more precise retireval at the cost of reduced contextual continuity and larger chunks which gives a richer context at the cost of increased retrieval noise.<br>
A moderate chunk size of 100 was selected to balance retrieval precision and contextual coverage.
#### Top-K Selection
The retrieval pipeline uses *top_k* = 3 retrieved chunks.
This value was intentionally fixed rather than dynamically tuned.
The considerations taken into account were:<br>
- too few chunks may omit relevant evidence
- too many chunks may dilute prompts with irrelevant context
- larger retrieval windows increase token usage and latency<br>
Top-3 retrieval provided a practical balance between evidence coverage and prompt efficiency.
#### Lexical Matching for Retrieval Metrics
Retrieval metrics:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)<br>
are computed using lexical overlap between retrieved chunks and ground-truth answers.
This approach was chosen because it is deterministic, interpretable, simple to benchmark and computationally lightweight.<br>
However, lexical matching may under-report semantically relevant retrieval when wording differs from the reference answer. This limitation was intentionally accepted in exchange for transparent and reproducible retrieval evaluation.
### 5.6 Sequential Inference
Generation and evaluation currently run sequentially. This design was chosen because it provides simpler orchestration and managebale rate limiting along with easier debugging.<br>
**Tradeoffs:** The primary limitation is increased runtime when evaluating larger datasets though concurrent API orchestration remains a possible future optimization option.
### 5.7 FastAPI & Streamlit Architecture
The deployment stack separates FastAPI backend and Streamlit frontend. This design follows separation-of-concerns principles.<br>
**FastAPI** handles API routing, request validation, evaluation orchestration and execution. 
Whereas, **Streamlit** handles the file uploads, dashboard presentation, user interaction and result visualization.<br>
This architecture keeps inference logic independent from presentation logic and simplifies deployment and maintenance.
### 5.8 Tradeoffs & Constraints
Several practical constraints shaped the system design:
- the framework currently benchmarks QA tasks only
- public deployment limits evaluation datasets to 10 samples
- transformer loading may introduce cold-start latency
- retrieval metrics rely on lexical overlap
- OpenAI API usage introduces cost and rate-limit considerations<br>
These constraints were accepted to preserve deployment feasibility while maintaining transparent and interpretable evaluation behavior.

## 6. Output Structure & Results
This framework produces two levels of output:
1. **Sample-level evaluation results:** Individual comparisons between baseline and RAG-enhanced responses.
2. **Aggregate evaluation metrics:** Dataset-wide performance summaries across retrieval, semantic, lexical, and judge-based metrics.<br>
Outputs are generated in structured JSON format, making them suitable for both manual inspection and downstream analysis.

### 6.1 Evaluation Output Schema
Example:
![Example Sample-level Output Schema](assets/Sample-level%20Output%20SS.png)<br>
This structure preserves both retrieval evidence and response-level evaluation, allowing transparent inspection of model behavior.
### 6.2 Aggregate Metrics
Aggregate results summarize overall framework performance across the evaluation dataset.
Example results from a 150-sample evaluation run:
| Metric              | Baseline |  RAG |
| ------------------- | -------: | ---: |
| Exact Match (EM)    |     0.00 | 0.00 |
| Semantic Similarity |     0.31 | 0.49 |
| Correctness         |     3.32 | 4.51 |
| Completeness        |     3.42 | 4.44 |
| Fluency             |     4.87 | 4.98 |

Retrieval performance:
| Retrieval Metric | Score |
| ---------------- | ----: |
| Precision@K      |  0.34 |
| Recall@K         |  0.95 |
| MRR              |  0.91 |

These results indicate consistent improvements in semantic alignment and qualitative response quality under retrieval augmentation.
### 6.3 Sample-Level Comparisons
The framework supports direct comparison between baseline and retrieval-augmented outputs.
#### Example 1 — Retrieval Improvement
**Question**
```text
What is the economy of Albion?
```
**Baseline**
```text
I don't know.
```
**RAG**
```text
The economy of Albion Online is fully player driven.
```
**Retrieved Evidence**
```text
...The game has a fully player driven economy...
```
In this example, semantic retrieval surfaced relevant supporting evidence that enabled the retrieval-augmented system to generate a grounded and complete answer.
#### Example 2 — Retrieval Success but Metric Limitation
**Question**
```text
What kind of game is Genshin?
```
**Ground Truth**
```text
fee-to-play gacha game
```
**RAG Output**
```text
...a free-to-play game monetized through gacha mechanics...
```
Despite strong retrieval and qualitatively correct generation, comparing against intentionally typo-ed ground-truth answer, Exact Match and lexical retrieval metrics remained conservative due to phrasing differences.<br>
This highlights the importance of combining lexical, semantic, and qualitative evaluation strategies.
### 6.4 Dashboard & Visualization
The deployed Streamlit interface presents:
- aggregate metrics
- metadata
- per-sample comparisons
- downloadable JSON outputs<br>
*(try it for yourself at https://llmragbenchmarking.streamlit.app/)*<br>
This visualization layer allows users to inspect evaluation behavior interactively while preserving access to raw machine-readable results.
### 6.5 Result Interpretation
Across evaluation runs, retrieval augmentation generally improved:
- semantic similarity
- factual correctness
- response completeness<br>
all while maintaining high fluency.<br>
However, evaluation also revealed that retrieval success does not always guarantee stronger generation and that lexical metrics alone may under-represent semantically correct outputs.<br>

## 7. Observations & Failure Cases
### 7.1 Key Observations
- Retrieval augmentation generally improved semantic similarity and qualitative response quality.
- RAG responses showed stronger correctness and completeness compared to baseline generation.
- Fluency remained consistently high across both baseline and retrieval-augmented outputs.
- Retrieval quality and generation quality were related but not always directly coupled.<br>
### 7.2 Failure Cases
**1. Retrieval failure:** When relevant evidence was not retrieved, RAG performance degraded and frequently defaulted to abstention responses such as *I don't know.* This highlighted the dependence of generation quality on retrieval quality.<br>
**2. Retrieval success didn't guarantee generation success:** Some cases retrieved relevant context successfully but still produced weak or incomplete answers, demonstrating that strong retrieval alone does not guarantee optimal downstream generation.<br>
**3. Lexical Metrics:** Exact Match and lexical retrieval metrics occasionally under-scored semantically correct outputs due to phrasing differences.<br>
These observations reinforced the importance of combining lexical, semantic, retrieval, and qualitative evaluation strategies when benchmarking RAG systems.
## 8. Future Improvements
The current framework establishes a functional baseline for benchmarking retrieval-augmented QA systems with possiblities of further developments and improvements.<br>
Potential future directions include:
- Concurrent API orchestration to reduce evaluation runtime
- Expanded evaluation datasets and larger benchmarking runs 
- Experimentation with alternative retrieval strategies and embedding models
- Broader benchmarking across additional NLP tasks such as machine translation or summarization<br>
These improvements are intended as incremental extensions rather than major architectural changes, preserving the framework's emphasis on interpretable evaluation.
## 9. Conclusion
This project is an attempt to design and implement an end-to-end NLP evaluation system focused on Retrieval-Augmented Generation benchmarking. Beyond response generation itself, the framework emphasizes measurable evaluation, modular engineering, and transparent analysis of retrieval behavior through multiple complementary metrics. The resulting system combines local experimentation, deployed inference, and structured evaluation into a single workflow, serving as a practical exploration of how retrieval and generation systems can be evaluated in a more systematic and reproducible manner.







