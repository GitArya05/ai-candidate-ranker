# AI Candidate Matcher & Discovery Ranker Engine

An end-to-end, high-throughput semantic search and context-aware alignment pipeline designed to parse, index, and rank large-scale candidate pools against target job descriptions. Optimized to execute entirely on cost-effective CPU infrastructure, this architecture guarantees zero external API dependencies and native resilience against adversarial keyword-stuffing hacks.

###  Project Assets
* **Production Cloud Sandbox (Hugging Face Spaces):** [Live Interactive Environment](https://huggingface.co/spaces/a-urkude/Candidate-Ranker)
* **Execution Command:** `python src/main.py`

---

##  System Architecture & Execution Flow

The system implements a production-grade **Two-Stage Retrieval & Reranking (Bi-Encoder + Cross-Attention) Architecture** to balance massive scale with granular contextual precision.


```text
+--------------------------+      +--------------------------+
|  Target Job Description  |      |   100K+ Candidate Pool   |
|         (.docx)          |      |         (.jsonl)         |
+------------+-------------+      +------------+-------------+
             |                                 |
             v                                 v
+------------------------------------------------------------+
|             STAGE 1: Dense Semantic Retrieval              |
|  - Sentence-Transformers (all-MiniLM-L6-v2)                |
|  - FAISS Vector Database Clustering                        |
|  - High-Throughput Matrix Stream Caching                   |
+----------------------------+-------------------------------+
                             |
                             | [Top 100 Coarse Candidates]
                             v
+------------------------------------------------------------+
|             STAGE 2: Granular Entity Alignment             |
|  - Linguistic Information Extraction (spaCy NLP)           |
|  - Core Technical Competency Matching                      |
|  - Experience Duration Trust Multiplier                    |
|  - Honeypot & Keyword Stuffing Adversarial Penalty Filter  |
+----------------------------+-------------------------------+
                             |
                             v
               +---------------------------+
               | Final Top 100 Ranked List |
               |     (submission.csv)      |
               +---------------------------+


```

### 1. Stage 1: Dense Semantic Retrieval (Coarse Filter)
* **Text Extraction:** Raw unstructured data is ingested from target files (`.docx`, `.jsonl`). 
* **Vector Embedding Generation:** Text segments are translated into a compressed continuous vector space utilizing the `all-MiniLM-L6-v2` Bi-Encoder model, mapping inputs to precise 384-dimensional dense tensors.
* **FAISS Clustering Index:** Embeddings are written into a Facebook AI Similarity Search (`FAISS`) index cluster. An optimized inner-product (cosine similarity) scan performs rapid nearest-neighbor search, filtering down the $100\text{K}+$ candidate pool to the **Top 100 raw semantic candidates** in milliseconds.

### 2. Stage 2: Fine-Grained Entity Alignment (Reranker)
* **Linguistic Traversal:** The top 100 profiles go through a secondary text processing step using a pre-compiled `spaCy` NLP pipeline to isolate proper nouns, core skills, and explicit technologies.
* **Experience-Duration Trust Matrix:** Instead of performing simple bag-of-words counting, the algorithm evaluates skill depth against chronological experience years to generate a trust multiplier.
* **Adversarial Resiliency (Honeypot Protection):** Explicit constraints penalize anomalous keyword frequency distributions. Profiles that copy-paste the job description to cheat vector space closeness are detected and systematically demoted.

---

##  Technical Decisions & Rationale

| Technology Choice | Component Layer | Rationale & Trade-offs |
| :--- | :--- | :--- |
| **`all-MiniLM-L6-v2`** | Dense Embedding Model | Offers a perfect 384-dimension semantic footprint with 99% of the performance of heavy BERT models while running natively on CPU at $10\times$ the speed. |
| **`FAISS`** | Vector Index Storage | Leverages optimized C++ memory alignments for fast vector comparisons, completely avoiding the overhead of external microservice vector DBs. |
| **`spaCy`** | Rule-Based Alignment Layer | Provides high-speed tokenization and deterministic entity extraction to catch semantic vulnerabilities that pure vector distances overlook. |
| **Memory-Stream Caching** | Infrastructure Utility | Processes `.jsonl` entries line-by-line via continuous generators, preventing Out-Of-Memory (OOM) crashes even under tight hardware container bounds. |

---

##  Evaluation Benchmarks

* **Hardware Used for Validation:** Local Lenovo Laptop (8 CPU Cores, 16 GB RAM, No GPU).
* **Dataset Scale:** 100,000 Complete Candidate Profiles.
* **Inference Speed:** Ingests, clusters, matches, and logs outputs in **under 2 minutes** entirely on CPU.
* **Network Overhead:** 0 API Calls during the ranking phase (fully network-isolated).

---

##  How to Reproduce & Run Locally

### Prerequisites
Ensure you have Python 3.11 installed along with `pip`.

### 1. Clone the Repository
```bash
git clone [https://github.com/a-urkude/ai-candidate-ranker.git](https://github.com/a-urkude/ai-candidate-ranker.git)
cd ai-candidate-ranker

```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
### 3. Run the Pipeline Execution
```bash
python src/main.py
```
### 4. Output Artifacts
Upon completion, the final ranked candidates spreadsheet will be formatted and exported directly to your workspace:
```
Output Path: data/submission.csv
```




