# 🧬 LlamaAffinity – Antibody Affinity Prediction with LLama 3
---  
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)  ![Llama3](https://img.shields.io/badge/Llama3-886FBF?style=for-the-badge&logo=meta&logoColor=fff) ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white) ![Keras-Hub](https://img.shields.io/badge/Keras--Hub-gray?style=for-the-badge)

---
This repository provides an implementation of **antibody affinity prediction** using a **LLaMA3 transformer backbone** with **5-fold cross-validation**. The model is trained and evaluated on protein molecular sequence OAS data procuring from AntiFormer github  and benchmarked and assessed with multiple metrics, including **accuracy, precision, recall, F1-score, ROC-AUC**, and training time.  

---

## 🚀 Project Information 
 
- **5-fold stratified cross-validation** for robust evaluation.  
- Model built using **keras-hub LLaMA3Backbone**.  
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC.  
- Visualizations: Confusion Matrix, ROC Curve, Loss/Accuracy plots.  
- Model saving and checkpointing for each fold.  

---

## ⚙️ Requirements

* Python 3.10+
* TensorFlow 2.15+
* keras-hub, keras-nlp
* HuggingFace datasets
* RDKit
* scikit-learn, matplotlib, seaborn
---

## 📂 Data Acquisition
```bash

AntiFormer/
├── data/                   # Dataset & processing scripts
│   ├── cdr1H.txt
│   ├── cdr1L.txt
│   ├── cdr2H.txt
│   ├── cdr2L.txt
│   ├── cdr3H.txt.zip
│   ├── cdr3L.txt
│   ├── manifest\_230324.csv
│   ├── data\_download.py
│   ├── data\_process.py
│   ├── dataset\_making.py
│   ├── dt\_rebuild.py
│   └── protbert/           # Tokenizer files for protein language model
│
├── subdt/                  # HuggingFace preprocessed dataset
│   ├── data-00000-of-00001.arrow
│   ├── dataset\_info.json
│   └── state.json


````

---

## 📦 Installation  

Clone the AntiFormer and LlamaAffinity repositories and install dependencies:  

```bash
git clone https://github.com/QSong-github/AntiFormer.git
cd AntiFormer

pip install -q rdkit datasets keras-nlp keras-hub tensorflow matplotlib seaborn scikit-learn
````

---

## 📊 Usage

### 1. Load Dataset from AntiFormer repo

The dataset is stored in HuggingFace format:

```python
from datasets import load_from_disk
dataset = load_from_disk('subdt')
```

### 2. Preprocess & Split Folds

Stratified K-Fold splitting ensures balanced label distribution:

```python
from sklearn.model_selection import StratifiedKFold
```

### 3. Model Training (5-fold CV)

```python
model = get_model()
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(0.0001),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
history = model.fit(ds_train, epochs=10, validation_data=ds_test)
```

### 4. Evaluation & Metrics

* Confusion Matrix
* ROC Curve & AUC
* Accuracy & Loss trends

### 5. Save & Export

Weights for each fold are saved to:

```
saved_models/model_fold_{fold}.weights.h5
```

---

## 📈 Results

* Cross-validation results of the LlamaAffinity model across five folds:

| Fold | Accuracy | F1 Score | Precision | Recall | ROC AUC | Training (Minutes) |
|------|----------|----------|-----------|--------|---------|---------------------|
| 0    | 0.9550   | 0.9548   | 0.9694    | 0.9406 | 0.9886  | 5.9600              |
| 1    | 0.9525   | 0.9533   | 0.9510    | 0.9557 | 0.9913  | 4.9559              |
| 2    | 0.9675   | 0.9674   | 0.9847    | 0.9507 | 0.9961  | 5.9042              |
| 3    | 0.9725   | 0.9728   | 0.9752    | 0.9704 | 0.9969  | 5.4559              |
| 4    | 0.9725   | 0.9730   | 0.9706    | 0.9754 | 0.9951  | 5.2030              |
| **Average** | **0.9640** | **0.9643** | **0.9702** | **0.9586** | **0.9936** | **27.4790** |

* Comparison of antibody affinity prediction models across performance metrics and training time  

| Model            | Accuracy | F1 Score | Precision | Recall | ROC AUC | Training (hours) |
|------------------|----------|----------|-----------|--------|---------|------------------|
| Transformer-6 L  | 0.7865   | 0.7590   | 0.8060    | 0.7990 | 0.7930  | 0.38             |
| Transformer-12 L | 0.8011   | 0.7890   | 0.8310    | 0.8180 | 0.8290  | 0.63             |
| AntiBERTy        | 0.8321   | 0.8510   | 0.9110    | 0.8910 | 0.9400  | 1.46             |
| AntiBERTa        | 0.8796   | 0.8570   | 0.9080    | 0.9090 | 0.9340  | 2.97             |
| AntiFormer       | 0.9169   | 0.8820   | 0.9630    | 0.9250 | 0.9660  | 0.76             |
| **LlamaAffinity**| **0.9640** | **0.9643** | **0.9702** | **0.9586** | **0.9936** | **0.46** |

---

## 📉 Visualizations

* ✅ Confusion Matrix
* ✅ ROC Curve
* ✅ Loss & Accuracy plots

---

---
## 📌 Citation

If you use this repository in your research, please cite:

```bibtex
@misc{<fillup>,
  author       = {Md Delower Hossain, Jake chen},
  title        = {LlamaAffinity: a compact sequence-only model for antibody binder propensity},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/aimed-lab/LlamaAffinity/blob/main/LlamaAffinity.ipynb}
}
```

---

## 🤝 Acknowledgements

* [QSong-github/AntiFormer](https://github.com/QSong-github/AntiFormer) for the original repo.
* HuggingFace `datasets` for easy dataset handling.
* keras-hub team for the LLaMA3 backbone implementation.
* Llama 3


---
