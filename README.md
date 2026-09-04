# Explainable Transfer Learning for Brain Tumor MRI Classification
> **A Four-Model Benchmark with Duplicate-Controlled Internal Testing and External Validation**

[![Paper PDF](https://img.shields.io/badge/Paper-Download%20PDF-red?style=for-the-badge&logo=adobeacrobatreader)](paper/Explainable_Brain_Tumor_MRI_Classification_Paper.pdf)
[![Poster PDF](https://img.shields.io/badge/Poster-Download%20PDF-blue?style=for-the-badge&logo=adobeacrobatreader)](paper/Explainable_Brain_Tumor_MRI_Classification_Poster.pdf)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org/)

---

## 📌 Publication Details

- **Course**: Computer Vision and Pattern Recognition [D] (Spring 25-26)
- **Department**: Department of Computer Science, Faculty of Science and Technology
- **Institution**: American International University-Bangladesh (AIUB)
- **Authors**:
  - **Mehedi Hasan Rafid** ([22-48453-3@student.aiub.edu](mailto:22-48453-3@student.aiub.edu))
  - **Al-Mahmud Zaman** ([22-49497-3@student.aiub.edu](mailto:22-49497-3@student.aiub.edu))
  - **M. R. Wasik Ahmed Apon** ([22-47698-2@student.aiub.edu](mailto:22-47698-2@student.aiub.edu))

---

## 📖 Abstract

Brain tumor classification from magnetic resonance imaging (MRI) is a clinically important computer vision problem, but many public-dataset studies rely only on internal test accuracy and do not sufficiently control image duplication or external generalization. This paper presents an explainable four-class brain tumor MRI classification benchmark for **glioma, meningioma, no-tumor, and pituitary** categories.

The internal development cohort was constructed from the Masoud Nickparvar and Ishans24 Kaggle datasets, while the Mendeley 2026 Brain Tumor MRI Dataset was held out as an independent external validation source. A preprocessing and leakage-control pipeline detected **28,908 raw images**, identified **20,400 duplicate entries**, removed duplicate overlap, and produced **15,721 final images with zero duplicate rows** across train, validation, internal test, and external validation splits.

Four models were evaluated under a unified PyTorch protocol: a scratch-trained **Basic CNN** and three ImageNet-initialized transfer-learning models, namely **ResNet50**, **EfficientNetB0**, and **DenseNet121**. **ResNet50** achieved the highest internal test accuracy (**98.49%**), while **DenseNet121** achieved the strongest external validation performance (**87.79% accuracy, 87.95% weighted F1-score**). Grad-CAM visualizations confirmed that model predictions were supported by tumor-relevant brain regions.

---

## 📊 Benchmark Results

### 1. Model Comparison (Internal Test vs. External Validation)

| Model | Training Strategy | Int. Acc (%) | Int. W-F1 (%) | Ext. Acc (%) | Ext. Prec (%) | Ext. Rec (%) | Ext. W-F1 (%) | Main Conclusion |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Basic CNN** | Scratch CNN Baseline | 96.16% | 96.18% | 77.78% | 78.52% | 77.78% | 77.65% | Largest external accuracy drop |
| **ResNet50** | ImageNet TL + Fine-Tune | **98.49%** | **98.49%** | 87.42% | 89.47% | 87.42% | 87.58% | **Best Internal Accuracy** |
| **EfficientNetB0** | ImageNet TL + Fine-Tune | 97.99% | 97.99% | 78.72% | 84.41% | 78.72% | 78.94% | Sensitive to domain shift |
| **DenseNet121** | ImageNet TL + Fine-Tune | 96.60% | 96.63% | **87.79%** | **88.62%** | **87.79%** | **87.95%** | **Best External Generalization** |

### 2. External Validation Class-Wise F1-Scores

| Model | Glioma F1 (%) | Meningioma F1 (%) | No-Tumor F1 (%) | Pituitary F1 (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Basic CNN** | 77.90% | 70.26% | 84.40% | 76.42% |
| **ResNet50** | 89.24% | **86.96%** | 85.14% | 88.13% |
| **EfficientNetB0** | 82.49% | 75.38% | 78.27% | 77.41% |
| **DenseNet121** | **89.10%** | 80.34% | **90.00%** | **90.27%** |

---

## 🔍 Key Methodological Insights

1. **Duplicate Control is Essential**: Naive splitting without deduplication inflates performance due to repeated MRIs across splits. Filtering out 20,400 duplicate rows produced a reliable, leakage-free evaluation.
2. **Domain Shift Changes Model Ranking**: **ResNet50** outperforms all models internally, but **DenseNet121** generalizes best on independent external data (Mendeley 2026).
3. **Explainability with Grad-CAM**: Gradient-weighted Class Activation Mapping confirms that DenseNet121 and ResNet50 focus on tumor-proximal regions rather than background artifacts.

---

## 📁 Repository Structure

```
cvpr/
├── paper/
│   ├── Explainable_Brain_Tumor_MRI_Classification_Paper.pdf     # Full 12-page IEEE Paper
│   └── Explainable_Brain_Tumor_MRI_Classification_Poster.pdf    # Conference Poster
├── notebooks/
│   ├── 01_dataset_preprocessing.ipynb                           # Preprocessing & deduplication
│   ├── 02_classification_training.ipynb                         # Model training (ResNet, DenseNet, etc.)
│   ├── 03_xai_gradcam.ipynb                                     # Grad-CAM heatmap generation
│   ├── 04_monai_segmentation.ipynb                              # MONAI segmentation experiment
│   ├── 05_results_analysis_and_paper_assets.ipynb              # Evaluation metrics & plots
│   └── CNN_22-48453-3.ipynb                                     # Custom CNN baseline notebook
├── .gitignore
└── README.md
```

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/mehedihasanrafid/cvpr.git
   cd cvpr
   ```

2. **Install Dependencies**:
   ```bash
   pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn opencv-python Pillow
   ```

3. **Run Notebooks**:
   Open any notebook inside the `notebooks/` directory using Jupyter Notebook, JupyterLab, or Google Colab.


