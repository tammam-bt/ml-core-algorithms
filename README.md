# 🧠 Machine Learning From Scratch

A curated collection of classical machine learning algorithms implemented **entirely from scratch** using Python and NumPy.

No `scikit-learn`.
No `tensorflow`.
No black boxes.

This repository focuses on **mechanical understanding**, not API usage.

---

## 🎯 Motivation

Most machine learning resources teach how to *use* models, not how they *work*.

This project exists to:

* Expose the full training loop of each algorithm
* Make optimization and loss functions explicit
* Reveal numerical and statistical failure modes
* Build intuition through visualization and experimentation

If you can implement it from scratch, you understand it.

---

## 📚 Algorithms Implemented

### 🧪 Supervised Learning

#### Regression

* Linear Regression (Normal Equation)
* Linear Regression (Gradient Descent)
* Ridge Regression
* Lasso Regression
* Polynomial Regression

#### Classification

* Logistic Regression
* k-Nearest Neighbors (k-NN)
* Naive Bayes (Gaussian)
* Perceptron
* Support Vector Machine (Linear)

---

### 🔍 Unsupervised Learning

* K-Means Clustering
* Principal Component Analysis (PCA)
* Hierarchical Clustering (Agglomerative)

---

### ⚙️ Optimization Algorithms

* Batch Gradient Descent
* Stochastic Gradient Descent
* Mini-batch Gradient Descent

---

### 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Mean Squared Error (MSE)
* Cross-Entropy Loss

---

## 🗂️ Project Structure

```
ml-from-scratch/
│
├── algorithms/
│   ├── regression/
│   │   ├── linear_regression.py
│   │   ├── logistic_regression.py
│   │   └── ridge_lasso.py
│   │
│   ├── classification/
│   │   ├── knn.py
│   │   ├── naive_bayes.py
│   │   ├── perceptron.py
│   │   └── svm.py
│   │
│   └── unsupervised/
│       ├── kmeans.py
│       ├── pca.py
│       └── hierarchical.py
│
├── optimization/
│   ├── gradient_descent.py
│   └── loss_functions.py
│
├── metrics/
│   └── evaluation.py
│
├── utils/
│   ├── data_utils.py
│   └── visualization.py
│
├── notebooks/
│   ├── regression_demo.ipynb
│   ├── classification_demo.ipynb
│   └── clustering_demo.ipynb
│
├── tests/
│   └── test_models.py
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🧩 Design Philosophy

* **Explicit over implicit** – every operation is visible
* **Readable over clever** – clarity beats abstraction
* **Educational over optimized** – performance is secondary
* **Deterministic behavior** – controlled randomness where applicable

---

## 🔎 Example: Logistic Regression

Each implementation includes:

* Manual sigmoid computation
* Binary cross-entropy loss
* Gradient derivation and update
* Decision threshold tuning
* Confusion matrix analysis
* Visualization of predictions

Nothing is hidden behind helper libraries.

---

## 🛠️ Technologies & Dependencies

### Core Technologies

* 🐍 **Python**
* 📓 **Jupyter Notebooks**

### Dependencies (Minimal by design)

```
numpy
matplotlib
```

Optional (for notebooks):

```
jupyter
```

---

## 🚀 Installation

```bash
git clone https://github.com/your-username/ml-from-scratch.git
cd ml-from-scratch
pip install -r requirements.txt
```

---

## ▶️ How to Use

* Run individual `.py` scripts directly
* Explore algorithm behavior through Jupyter notebooks
* Modify loss functions, learning rates, or initialization to observe effects

This repository is designed for experimentation.

---

## 👥 Intended Audience

* Students learning machine learning fundamentals
* Engineers preparing for ML interviews
* Practitioners who want to understand models beyond library calls

This is not a production ML framework.

---

## ⚠️ Known Limitations

* Not optimized for large-scale datasets
* No GPU acceleration
* No deep learning models

These constraints are intentional.

---

## 📄 License

MIT License — free to use, modify, and distribute for educational purposes.
