# 📊 Sentiment Analysis Using NLP (AI341 Project)

This project is a sentiment analysis system that classifies movie reviews as **positive** or **negative** using **Natural Language Processing (NLP)** and **Deep Learning (LSTM)**. It was developed as part of the AI341 course at university.

---

## 👨‍💻 Team Members

| Name                    | Student ID     |
|-------------------------|----------------|
| Shady Wafik Heshmat     | 200026518      |
| Adham Eloraby           | 200029570      |
| Mostafa Farag           | 200034359      |
| Yousef Mohamed Ali      | 200033377      |
| Yousef Mohamed Salama   | 200033523      |

---

## 📂 Project Overview

This project aims to build and train a sentiment classifier based on a dataset of movie reviews. It uses both classical NLP techniques and modern deep learning (Bi-LSTM + Embeddings) for binary classification.

---

## 📦 Libraries Used

- **Data Handling**: `pandas`, `numpy`
- **Preprocessing**: `nltk`, `re`
- **Feature Extraction**: `CountVectorizer`, `TfidfVectorizer`
- **Embeddings**: `SentenceTransformer`, `Tokenizer`, `pad_sequences`
- **Modeling**: `Sequential`, `LSTM`, `Dense`, `Adam` from TensorFlow/Keras
- **Evaluation**: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`
- **Utilities**: `train_test_split`, `EarlyStopping`, `ReduceLROnPlateau`

---

## 🧹 Data Preprocessing

- **Cleaning**: Removed HTML tags, URLs, special characters, and stopwords
- **Normalization**: Lowercasing, tokenization, stemming, lemmatization
- **New Column**: Created `cleaned_review` for the preprocessed text

---

## 📊 Dataset Summary

- **Total Reviews**: 1,999  
- **Sentiment Labels**:  
  - Positive: 1,005  
  - Negative: 994  
- **Duplicates**: 0  
- **Missing Values**: None  
- **Balanced Dataset**: Yes ✅

---

## 🔠 Feature Engineering

- **Bag of Words (BoW)**  
  - Shape: `(1999, 17155)`
- **TF-IDF**  
  - Shape: `(1999, 17155)`
- **N-Grams (Bi-grams)**  
  - Shape: `(1999, 177377)`
- **Sentence Embeddings**  
  - Model: `paraphrase-MiniLM-L6-v2`  
  - Shape: `(1999, 384)`

---

## 📚 Deep Learning Model

- **Tokenizer**: Top 5,000 words  
- **Sequence Length**: 500 (with padding)  
- **Architecture**:
  - Embedding Layer: 128 dimensions  
  - Bidirectional LSTM: 64 units  
  - Dropout: 0.5  
  - Output: 1 neuron, `sigmoid` activation

---

## 🏋️ Model Training

- **Optimizer**: Adam (lr=0.001)  
- **Loss**: Binary Crossentropy  
- **Batch Size**: 32  
- **Epochs**: Up to 15  
- **EarlyStopping**: Patience = 3  
- **ReduceLROnPlateau**: Patience = 2

---

## ✅ Model Performance

| Metric     | Score    |
|------------|----------|
| Accuracy   | 84%      |
| Precision  | 84%      |
| Recall     | 85%      |
| F1-Score   | 85%      |
