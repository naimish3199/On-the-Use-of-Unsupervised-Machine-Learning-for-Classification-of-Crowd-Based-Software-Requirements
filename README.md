# On-the-Use-of-Unsupervised-Machine-Learning-for-Classification-of-Crowd-Based-Software-Requirements

This repository contains the source code associated with the paper titled **On-the-Use-of-Unsupervised-Machine-Learning-for-Classification-of-Crowd-Based-Software-Requirements**. The code facilitates the reproduction of the experiments and results presented in the paper.

## Prerequisites

- Ensure that Python 3.11 or higher is installed on your system.

## Running the Project Locally

- **Clone the repository**

  ```bash
  git clone https://github.com/naimish3199/On-the-Use-of-Unsupervised-Machine-Learning-for-Classification-of-Crowd-Based-Software-Requirements.git
  ```

- **Navigate to the Project Directory**
  ```bash
  cd <path_to_cloned_repository>
  ```
- **Set Up a Virtual Environment (Recommended)**

  Python’s built-in venv module can be used as follows:

  - **Windows**

    ```bash
    python -m venv newenv
    newenv\Scripts\activate
    ```

  - **Linux/macOS**
    ```bash
    python3 -m venv newenv
    source newenv/bin/activate
    ```

- **Install Required Dependencies**

  Once the virtual environment is activated, install the necessary dependencies:

  ```bash
  pip install -r requirements.txt
  ```

## Usage

- **MRR and NDCG calculation:**

  To evaluate the performance of various embedding techniques using Mean Reciprocal Rank (MRR) and Normalized Discounted Cumulative Gain (NDCG) metrics, execute the `mrr_ndcg_calculation.ipynb` notebook.

As outlined in the paper, after determining the top 3 performing embeddings on CrowdRE dataset, subsequent experiments were conducted using these top embeddings.

- **Embeddings Utilized**

  The following embedding techniques were employed in the experiments:

  - **SRoBERTa** (https://huggingface.co/sentence-transformers/all-distilroberta-v1)
  - **SBERT** (https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2)
  - **Word2Vec** (Trained on CrowdRE dataset)

- **Clustering Techniques**

  The clustering methods used in this study include:

  - **K-means**
  - **Hierarchical Agglomerative Clustering (HAC)**

- **Labelling Methods**

  Labelling for the clusters was performed through both manual and automatic techniques:

  - **Manual Labelling**: Performed using **BERTopic**.
  - **Automatic Labelling**: Based on **Semantic Similarity**.

- **Number of clusters**

  The experiments were conducted with varying cluster numbers i.e. [2, 3, 4, 5].

- **Running the Experiments**

To replicate the experiments, execute the `main.py` script located in the `src/` folder. This script requires several command-line arguments to configure the clustering process for software requirements.

**Command-Line Arguments:**

| Argument       | Type  | Choices                            | Required | Description                                                                          |
| -------------- | ----- | ---------------------------------- | -------- | ------------------------------------------------------------------------------------ |
| `--mode`       | `int` | `0, 1`                             | ✅       | `0` for manual labeling, `1` for automated labeling.                                 |
| `--embedding`  | `int` | `0, 1, 2`                          | ✅       | Embedding technique: `0` - Word2Vec, `1` - SBERT, `2` - SRoBERTa.                    |
| `--clustering` | `int` | `0, 1`                             | ✅       | Clustering method: `0` - K-Means, `1` - Hierarchical Agglomerative Clustering (HAC). |
| `--merge`      | `int` | `0, 1`                             | ✅       | `1` - Merge Health and Other domains, `0` - No merge.                                |
| `--clusters`   | `int` | (2-4 if merged, 2-5 if not merged) | ✅       | Number of clusters to create.                                                        |

### **Example Usage**

To run automatic labeling with SRoBERTa embeddings, K-Means clustering, no merging of Health and Other domains, and 3 clusters, use the following command

```bash
python main.py --mode 1 --embedding 2 --clustering 0 --merge 1 --clusters 3
```
