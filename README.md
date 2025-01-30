# Polish Tweet Sentiment Classification with BERT

[![Work in Progress](https://img.shields.io/badge/status-WIP-yellow)](https://github.com/yourusername/your-repo)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This project explores sentiment classification of Polish tweets using BERT-based models. The current implementation focuses on handling class imbalance and optimizing preprocessing techniques.

**⚠️ Note: This is an active research project and work in progress. The codebase and results are subject to change.**

## 📌 Key Features
- **Class Imbalance Handling** (1400 neutral vs 250 positive/negative samples)
- Multiple Text Preprocessing Strategies
- Hyperparameter Grid Search

## 📂 Dataset Structure
```text
Class Distribution:
- Neutral: 1400 samples
- Positive: 250 samples
- Negative: 250 samples

Features:
- Raw tweet text
- Preprocessed versions
- Sentiment labels (0: negative, 1: neutral, 2: positive)
