# Term Deposit Subscription Prediction (Bank Marketing)

## Project Overview
This project analyzes customer data from a marketing campaign and builds a machine learning model to predict whether a customer will subscribe to a term deposit product.

## Dataset
The dataset comes from the **UCI Bank Marketing dataset** and includes two files:

- `bank-additional-full.csv`: used for **training**
- `bank-additional.csv`: used for **testing**

Each record includes demographic, financial, and previous campaign contact details for each customer.

## Project Workflow

### Data Preprocessing
- Remove duplicates and overlapping records between training and testing datasets.
- Separate features (`X`) and target labels (`y`).
- Encode target labels (`yes → 1`, `no → 0`).

### Handling Class Imbalance
- Use **SMOTEN** (Synthetic Minority Over-sampling Technique for Nominal features) to oversample the minority class.

### Pipeline Construction
- Use `Pipeline` and `ColumnTransformer` for:
  - Imputation of missing values
  - Standard scaling
  - One-hot encoding for categorical features
  - Ordinal encoding where appropriate

### Model Training
- Apply machine learning models such as:
  - `LGBMClassifier` (LightGBM)
  - `SGDClassifier` (Stochastic Gradient Descent)

### Hyperparameter Tuning
- Use `GridSearchCV` for optimal model parameter selection.

### Model Evaluation
- Metrics used: Accuracy, Precision, Recall, F1-score.
- Performance is tested on the held-out test dataset.

## Libraries Used
- `pandas`, `numpy`
- `scikit-learn`
- `lightgbm`
- `imblearn` (for SMOTEN oversampling)
- `ydata_profiling` (for exploratory data analysis)

## Project Structure
bank+marketing/
├── bank+marketing_dataset/        # Folder containing dataset files
├── final.ipynb                    # Main notebook containing full ML pipeline
├── output_test.html               # Test dataset detailed output
├── output_train.html              # Training dataset detailed output
├── requirements.txt               # Python dependencies
└── README.md                      # Project description

## Objective & Impact
This project aims to help financial institutions:
- Identify high-potential customers
- Increase the effectiveness of direct marketing campaigns
- Reduce costs by focusing on the right audience

## 🚀 How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
2. Run the notebook
    jupyter notebook final.ipynb

### LightGBM Model Evaluation
- **Best Parameters** for LGBMClassifier:  
  `{'classifier__learning_rate': 0.01, 'classifier__max_depth': -1, 'classifier__n_estimators': 300, 'classifier__num_leaves': 100}`

- **Best Recall on Train Set**:  
  `0.8838198101830468`

- **Test Set Results**:
          precision    recall  f1-score   support

       0       0.97      0.89      0.93      3668
       1       0.48      0.80      0.60       451
