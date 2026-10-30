# The Significance of Data Labeling and the Role of Snorkel in Enhancing Machine Learning

In the realm of machine learning (ML) and artificial intelligence (AI), data is the bedrock upon which models are built and refined. However, raw data, while abundant, often lacks the structure and organization necessary for effective learning. This is where data labeling comes into play. Data labeling is the process of annotating or tagging data points with relevant metadata or labels, providing context and meaning to otherwise unstructured information. This process is crucial for numerous reasons, and its importance cannot be overstated in the development and deployment of ML models.

- First and foremost, labeled data serves as the training material for ML algorithms. Supervised learning, one of the most prevalent paradigms in ML, relies heavily on labeled datasets to learn the relationship between input features and target outputs. Without accurately labeled data, ML models would struggle to generalize patterns and make accurate predictions. Thus, the quality and accuracy of labels directly influence the performance and reliability of ML systems.

- Furthermore, data labeling facilitates the creation of ground truth datasets, which serve as benchmarks for evaluating model performance. By comparing model predictions against accurately labeled data, ML engineers can assess the efficacy of their algorithms and identify areas for improvement. This iterative feedback loop is essential for refining ML models and enhancing their robustness in real-world scenarios.

- Additionally, labeled data enables domain-specific insights and knowledge extraction. By categorizing data points into meaningful classes or attributes, organizations can derive valuable insights about customer preferences, market trends, and business operations. These insights can inform strategic decision-making and drive innovation across various industries, from healthcare to finance to e-commerce.

Despite its undeniable importance, data labeling can be a labor-intensive and time-consuming process, especially when dealing with large volumes of unstructured data. Manual labeling by human annotators is often prone to errors, inconsistencies, and biases, leading to suboptimal model performance. This is where automated labeling tools like Snorkel come into play.

`Snorkel is a powerful framework designed to streamline the data labeling pipeline and mitigate the challenges associated with manual annotation. By leveraging weak supervision techniques, Snorkel enables ML teams to programmatically generate labels from noisy or imperfect sources, such as heuristics, rules, or distant supervision. This approach not only accelerates the labeling process but also improves label quality by aggregating information from multiple sources and learning from noisy signals.`

Moreover, Snorkel provides a unified platform for managing the entire data labeling workflow, from data ingestion to model training. Its flexible and extensible architecture allows users to customize labeling functions, integrate with existing ML pipelines, and adapt to evolving data requirements. By automating tedious labeling tasks and empowering ML engineers to focus on high-level model design and optimization, Snorkel enhances productivity and accelerates the pace of innovation in ML research and development.

In this tutorial we will learn fundamental aspects of Snorkel using 3 examples provided by Snorkel team.

# Spam Tutorials
We consider a canonical machine learning problem: classifying spam. This directory contains three tutorials, described below:
* `01_spam_tutorial`: This tutorial dives deep into how we can create, analyze, and use labeling functions for the spam classification task.
* `02_spam_data_augmentation_tutorial`: This tutorial demonstrates how to write, combine and apply transformation functions for performing data augmentation.
* `03_spam_data_slicing_tutorial`: This tutorial shows how we can use slicing functions to identify important slices of the data, for monitoring and improved performance.

# Dataset
Instead of the original YouTube Spam Collection, this lab uses five **synthetic YouTube comment datasets** in `data/`, each with the columns `COMMENT_ID, AUTHOR, DATE, CONTENT, CLASS` (`CLASS`: 0 = ham, 1 = spam). Many comments are Hinglish (Hindi written in Latin script), which motivates some of the custom labeling and slicing functions.

| File | Channel theme | Comments |
|---|---|---|
| `Youtube01-SyntheticBollywoodSong.csv` | Bollywood song | 360 |
| `Youtube02-SyntheticStandupComedy.csv` | Stand-up comedy | 380 |
| `Youtube03-SyntheticCookingChannel.csv` | Cooking channel | 410 |
| `Youtube04-SyntheticCricketHighlights.csv` | Cricket highlights | 430 |
| `Youtube05-SyntheticTechReview.csv` | Tech review | 370 |

`utils.py` (`load_spam_dataset`) loads them: files 01–04 are used for training, and file 05 is split into validation and test sets.

# Project Structure
```
mlops_lab3/
├── 01_spam_tutorial.ipynb                    # labeling functions + LabelModel
├── 02_spam_data_augmentation_tutorial.ipynb  # transformation functions
├── 03_spam_data_slicing_tutorial.ipynb       # slicing functions + slice-aware model
├── utils.py                                  # data loading, featurization, model helpers
├── data/                                     # synthetic YouTube comment CSVs
├── requirements.txt
└── README.md
```

# Setup
Python **3.10** is recommended (TensorFlow 2.15 and Snorkel 0.9.x are not compatible with the newest Python versions).

```bash
conda create -n mlops_lab3 python=3.10 -y
conda activate mlops_lab3
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m textblob.download_corpora
python -m ipykernel install --user --name mlops_lab3 --display-name "Python (mlops_lab3)"
```

Troubleshooting (seen on recent macOS on Apple Silicon):
- **`numpy.dtype size changed` when importing TensorFlow**: TensorFlow 2.15 needs NumPy 1.x, so install `numpy<2` (already pinned in `requirements.txt`).
- **`ImportError ... _spropack ... __thread_bss` when importing scikit-learn**: the pip SciPy wheel fails to load. Use the conda-forge build instead:
  `conda install -c conda-forge "numpy=1.26" "scipy=1.15" "libopenblas=*=*pthreads*"`
- **`OMP: Error #15 ... libomp.dylib already initialized` (kernel dies in notebook 03)**: two OpenMP runtimes were loaded (torch and OpenBLAS). Using the `pthreads` OpenBLAS build from the command above fixes it.

# How to Run
Open the notebooks in Jupyter Lab or VS Code, select the **Python (mlops_lab3)** kernel, and run them in order (01 → 02 → 03).

To execute all of them from the terminal and save the outputs into the notebooks:
```bash
for f in 0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=mlops_lab3 "$f"; done
```

# Results
Results from the latest run (numbers can vary slightly between runs because of random initialization).

### 01 – Labeling functions
Labeling functions used: keyword lookups (e.g. `check`/`check out`), regexes (`regex_check_out`, `regex_link_domains`, `regex_timestamp`), heuristics (`short_comment`), spaCy NER (`has_person_nlp`), and TextBlob polarity/subjectivity.

| Method | Test accuracy |
|---|---|
| Majority vote over LFs | 80.0% |
| Snorkel LabelModel | 72.8% |
| Classifier trained on LabelModel's probabilistic labels | **89.6%** |

### 02 – Data augmentation
Transformation functions: `change_person`, `swap_adjectives`, `replace_verb_with_synonym`, `replace_noun_with_synonym`, `replace_adjective_with_synonym`, combined with a `RandomPolicy` / `MeanFieldPolicy`.

| Training data | Test accuracy |
|---|---|
| Original | 53.2% |
| Augmented | 53.2% |

With this small synthetic dataset, the LSTM model is near chance and augmentation did not change the result.

### 03 – Data slicing
Slicing functions: `short_comment`, `keyword_please`, `regex_check_out`, `short_link`, `textblob_polarity`, plus two custom slices for this dataset: `money_scam` and `hinglish`.

Test set F1 (baseline model): 100%. Per-slice F1 for the slice-aware model:

| Slice | F1 |
|---|---|
| overall task | 0.777 |
| short_comment | 0.727 |
| keyword_please | 0.759 |
| regex_check_out | 0.867 |
| short_link | 1.000 |
| textblob_polarity | 0.000 |
| money_scam | 0.918 |
| hinglish | 0.512 |

The `hinglish` slice is the weakest meaningful slice, which shows how slicing functions expose subsets where the model underperforms even when the overall score looks good.

# Author
Hiral Rana ([@HiralR2931](https://github.com/HiralR2931))
