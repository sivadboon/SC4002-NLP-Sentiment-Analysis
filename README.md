# SC4002 NLP Sentiment Analysis
## Natural Language Processing Project - Nanyang Technological University

## Members
- Wang Shang An Davis (U2121998F)
- Li Zihan (U2121598G)
- Lye En Lih (U2121387B)
- Zhang Jing Wen (U2121853G)
- Isaiah Loh Kai En (U2140496L)
- Ong Yi Xin Kelly (U2040271D)

This project explores various deep learning architectures for sentiment analysis, optimizing word embeddings and RNN-based models to improve classification accuracy.

## Overview
This project implements word embeddings (GloVe, FastText) and deep learning models (RNN, BiLSTM, CNN, Attention) to classify sentiment in movie reviews. The models are benchmarked on test accuracy, with the best-performing architecture achieving 81.05% accuracy using a BiLSTM + BERT + Attention model.

## Features
**Preprocessing**: Tokenization, lemmatization, and handling out-of-vocabulary (OOV) words using FastText.
**Word Embeddings**: Comparison of GloVe and FastText embeddings for feature representation.
**Deep Learning Models**:
- RNN (Recurrent Neural Network)
- BiLSTM & BiGRU (Bidirectional LSTM/GRU)
- CNN (Convolutional Neural Networks)
- BERT + Attention Mechanism
  
**Hyperparameter Optimization**: GridSearch used to fine-tune learning rates, batch sizes, and model architectures.
**Evaluation Metrics**: Test accuracy, precision-recall, confusion matrix

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/SC4002_NLP_Sentiment_Analysis.git
cd SC4002_NLP_Sentiment_Analysis
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run Jupyter Notebook

```bash
jupyter notebook
```
