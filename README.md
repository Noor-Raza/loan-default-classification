# Loan Default Classification

This repository contains a single-notebook project focused on **classifying loan applications as safe vs risky**, with an emphasis on **business impact through cost-sensitive thresholding**.  
The project demonstrates how data science workflows can be aligned with business objectives, not just accuracy metrics.

---

## 📂 Repository Contents
- **`notebook.ipynb`** → Complete, end-to-end workflow (EDA → preprocessing → model training → evaluation → threshold optimization).
- **`report.pdf`** → Business report summarizing problem statement, modeling approach, and key findings.
- **`data/dataset.csv`** → Input dataset (tabular, anonymized).
- **`README.md`** → Documentation (this file).
- **`requirements.txt`** → Python dependencies for reproducibility.

---

## 🏦 Business Problem
Financial institutions need to decide whether to approve or reject loan applications.  
- **False Negatives (FN)** → Risky customers incorrectly classified as safe → leads to **loan defaults** (direct monetary loss).  
- **False Positives (FP)** → Safe customers incorrectly rejected → results in **lost revenue and customer dissatisfaction**.  

A good model must **minimize overall business cost**, not just maximize accuracy.

---

## 📊 Dataset
- Contains historical loan applications with borrower attributes and default outcomes.
- Imbalanced: **defaults (positives) are much rarer than non-defaults**.
- Target variable: `default` (1 = default, 0 = repaid).
- Features: demographics, financial ratios, credit behavior indicators (varies by dataset).

> The dataset is stored in `data/dataset.csv`.  
> For privacy, ensure sensitive information is anonymized.

---

## 🧪 Methodology
The project follows these steps:

1. **Exploratory Data Analysis (EDA)**  
   - Check distributions, correlations, and missing values.  
   - Identify imbalance in target classes.

2. **Data Preprocessing**  
   - Handle missing values, encode categorical features, scale numerical variables.  
   - Train/test split.

3. **Class Imbalance Handling**  
   - Apply **SMOTE (Synthetic Minority Oversampling Technique)** to balance classes.

4. **Model Training**  
   - **Random Forest** and **XGBoost** as primary classifiers.  
   - Hyperparameter tuning via cross-validation.

5. **Evaluation Metrics**  
   - Standard metrics: Accuracy, Precision, Recall, F1, ROC-AUC.  
   - Cost-sensitive metrics: total business cost using FP and FN penalties.

6. **Threshold Optimization**  
   - Instead of the default 0.5 probability cutoff, select a threshold that **minimizes business loss**:  
     ```
     total_cost = (FP × FP_COST) + (FN × FN_COST)
     ```
   - Grid search over thresholds between 0.1–0.9.  
   - Choose threshold with lowest expected cost.

---

## 📈 Results
- **Baseline (0.5 threshold):** good accuracy but higher business loss due to misclassified risky loans.  
- **Optimized threshold:** reduces expected loss by adjusting cutoff toward higher recall (catching more defaults).  
- **Random Forest vs XGBoost:**  
  - RF → stable, interpretable, slightly lower AUC.  
  - XGBoost → higher AUC, better calibrated probabilities, better performance after tuning.  

The **business-optimized threshold strategy** clearly outperforms default settings, showing the importance of aligning ML with financial objectives.

*(Exact metrics, confusion matrices, and plots are in `notebook.ipynb` and summarized in `report.pdf`.)*

---

## ⚙️ Installation & Usage
Clone the repository:

```bash
git clone https://github.com/<your-username>/loan-default-classification.git
cd loan-default-classification
