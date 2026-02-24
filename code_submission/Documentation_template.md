# ML Challenge 2025: Smart Product Pricing Solution Template

Team Name: *COYS*
Team Members: [SHASHANK PULLABHATLA, Varshith Reddy Amanaganti, Kaki Lahari Rohith]
Submission Date: [=13-10-2025]

---

## 1. Executive Summary

Our solution employs a *multi-modal ensemble* strategy combining the predictive power of a *Deep Neural Network (DNN)* with a *highly-tuned LightGBM* model. We extracted deep feature vectors from product text (DistilBERT) and images (ResNet50) and meticulously engineered structural features. The entire pipeline was optimized against the *Symmetric Mean Absolute Percentage Error (SMAPE)* to ensure high relative accuracy across the diverse product price range.

---

## 2. Methodology Overview

### 2.1 Problem Analysis
The challenge is a *Regression problem* evaluated by *SMAPE. The target variable (Price) is severely **right-skewed* (Mean $\approx \$23.65$, Median $\approx \$14.00$), necessitating a target transformation and models optimized for relative error.

Key Observations:
1.  *Skewness:* Required *$\mathbf{\log(\text{price} + 1)}$ transformation* to stabilize variance.
2.  *Structural Data:* The fixed Value: and Unit: fields allowed for reliable *Item Pack Quantity (IPQ)* extraction and *One-Hot Encoding (OHE)*.

### 2.2 Solution Strategy
We adopted a *Feature-Fusion Ensemble* approach. Features were concatenated into a single, high-dimensional vector before being passed to two powerful, cross-validated model types whose predictions were blended for the final output.

Approach Type: *Weighted Multi-Model Ensemble (LightGBM + DNN)*
Core Innovation: *Deep Feature Fusion* integrating transfer learning embeddings from *DistilBERT* (768D) and *ResNet50* (2048D) into a unified input, followed by *Optuna-based hyperparameter tuning* of the LightGBM model.

---

## 3. Model Architecture

### 3.1 Architecture Overview
The pipeline merges three parallel feature streams (Structural, Text, Image) into a single $\approx 2800$-dimensional scaled vector, which is used to train two separate models for ensembling.

*Feature Extraction Flow:*
1.  *Text $\to$ DistilBERT* (768D)
2.  *Image $\to$ ResNet50* (2048D)
3.  *Structural $\to$ IPQ/OHE* ($\approx$ 13D)
$\qquad\qquad\downarrow$
*Concatenation & Standardization*
$\qquad\qquad\downarrow$
*Ensemble:* (Tuned LGBM $\&$ 5-Fold DNN) $\to$ *Weighted Average* $\to$ Final Price

### 3.2 Model Components

| Component | Model/Technique | Key Features/Notes |
| :--- | :--- | :--- |
| *Structural* | *Feature Engineering* | Log-transformed IPQ, Title Length, Total Length. 10 OHE Unit categories. |
| *Text Pipeline* | *DistilBERT* | Extracted the *[CLS] token vector* (768D) from the frozen model. |
| *Image Pipeline* | *ResNet50* | Used frozen weights with *Global Average Pooling* to get the 2048D feature vector. |
| *LightGBM (GBM)* | *LGBMRegressor* | Hyperparameters optimized via *Optuna* to minimize validation SMAPE. |
| *Deep Neural Network* | *Keras Sequential (1024-512-256)* | Trained on GPU with *BatchNormalization* and *Early Stopping* for stability. |

---

## 4. Model Performance

The final prediction is the weighted average of the predictions from the two cross-validated models.

### 4.1 Validation Results
- SMAPE Score: *[1.8715%]*

---

## 5. Conclusion

Our approach successfully combined transfer learning, robust feature engineering, and advanced machine learning optimization. The two-model ensemble provides stability and accuracy, effectively blending the fast, structural power of GBMs with the non-linear feature discovery of the DNN, leading to a highly competitive final submission.

---