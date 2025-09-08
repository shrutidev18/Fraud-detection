# 🕵️ Unsupervised Credit Card Fraud Detection

## 📌 Problem Statement
Credit card fraud is a major issue for banks and payment systems. Fraudulent transactions are rare (~0.17% in this dataset), making detection difficult. Traditional supervised learning requires labeled fraud data, which is often unavailable in real-world scenarios.  
This project applies **unsupervised anomaly detection** to identify suspicious transactions without relying on labels.

---

## 🎯 Objectives
- Detect fraudulent transactions using **unsupervised ML techniques**.  
- Avoid **data leakage** by using time-based splits.  
- Build a beginner-friendly yet **resume-worthy project** that can later scale into an industry-ready solution.  

---

## 📊 Dataset
- Source: [Kaggle European Credit Card Transactions Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- ~284,807 transactions  
- Features: 28 anonymized PCA features (V1–V28), Time, Amount, Class (0=Normal, 1=Fraud).  
- Extremely **imbalanced** (frauds ≈ 0.17%).

---

## ⚙️ Tech Stack
- **Python 3**
- **Google Colab** (for development)
- **Libraries**:  
  - `pandas`, `numpy` → data manipulation  
  - `matplotlib`, `seaborn` → visualization  
  - `scikit-learn` → Isolation Forest, Local Outlier Factor  
  - `umap-learn`  → dimensionality reduction  
  - `gradio`  → tiny interactive demo

---

## 🔍 Approach
1. **Data Loading**: Kaggle dataset imported via Colab/Drive.  
2. **Minimal EDA**: Checked class imbalance, transaction time, and fraud percentages.  
3. **Time-based Split**: Train = first 70%, Test = last 30% → prevents leakage.  
4. **Downsampling**: Kept all fraud rows, sampled normal rows for faster training.  
5. **Feature Engineering**:  
   - Used anonymized V1–V28  
   - Robust-scaled `Amount`  
   - Added simple time features (`hour`, `night_flag`)  
6. **Models**:  
   - **Rule-based baseline** (night + high amount)  
   - **Isolation Forest** (main unsupervised model)  
   - **Local Outlier Factor (LOF)** for comparison  
7. **Evaluation**:  
   - Precision@K (top-K suspicious transactions)  
   - Recall@K  
   - PR-AUC (average precision)  
8. **Gradio Demo**: Simple app where user inputs amount/time → model outputs fraud likelihood.  

---

## 📈 Results (Mini Project)
- Isolation Forest detected **fraud transactions far above random chance**.  
- PR-AUC ≈ **0.87** (much better than random baseline).  
- Showed **top 20 suspicious transactions** for analysts.  

*(Insert a plot or table screenshot here → place in `reports/` and link)*

---

## 🚀 Future Improvements (Scaling to Major Project)
- Hybrid models (Autoencoder + Isolation Forest).  
- Explainability with SHAP/LIME.  
- Real-time fraud detection API (FastAPI + Docker).  
- Streaming pipeline (Kafka + Spark/Flink).  
- Graph-based fraud networks (Neo4j, PyTorch Geometric).  
- Deployment to cloud (AWS/GCP/Azure).  

---

## 📂 Repository Structure
Fraud-detection/
│── README.md
│── requirements.txt
│── .gitignore
│── notebook/
│     └── Fraud_Detection_Mini_Project.ipynb
│── reports/
│     └── sample_results.png
