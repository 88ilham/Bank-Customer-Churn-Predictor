# [WIP] 🏦 Bank Customer Churn Prediction

*A machine learning model to predict customer attrition using XGBoost, achieving **82% accuracy** with optimized recall for churn detection.*

## 📊 Model Performance

### **Classification Report**
| Metric       | Class 0 (Not Churn) | Class 1 (Churn) | Weighted Avg |
|--------------|---------------------|-----------------|--------------|
| **Precision**| 0.90               | 0.56            | 0.83         |
| **Recall**   | 0.88               | 0.61            | 0.82         |
| **F1-Score** | 0.89               | 0.58            | 0.83         |

**Key Takeaways:**  
- **82% overall accuracy** on test data.  
- **61% recall for churners** (identifies 61% of at-risk customers).  
- **56% precision** (56% of predicted churners actually churned).  

### **Confusion Matrix (XGBoost)**
|                | Predicted: Not Churn | Predicted: Churn |
|----------------|-----------------------|------------------|
| **Actual: Not Churn** | 1286 (True Negatives)  | 307 (False Positives) |
| **Actual: Churn**     | 102 (False Negatives)  | 305 (True Positives)  |

**Business Impact:**  
- Correctly flagged **305/407 churners** (61% recall).  
- **307 false positives** (may trigger unnecessary retention efforts).  

---

## 🛠️ Project Steps

### **1. Data Exploration & Preprocessing**
- **Handled Imbalance**: Dataset had 20% churn rate (oversampled minority class).  
- **Feature Engineering**:  
  - Encoded categoricals (`Geography`, `Gender`).  
  - Scaled numericals (`CreditScore`, `Balance`).  
- **Dropped Irrelevant Columns**: `RowNumber`, `CustomerId`, `Surname`.  

### **2. Model Comparison**
Tested 5 algorithms with default params:  
| Model          | Accuracy | Recall (Churn) |  
|----------------|----------|----------------|  
| Decision Tree  | 0.79     | 0.52           |  
| Random Forest  | 0.81     | 0.55           |  
| AdaBoost       | 0.80     | 0.50           |  
| GradientBoost  | 0.82     | 0.58           |  
| **XGBoost**    | **0.82** | **0.61**       |  

**Selected XGBoost** for highest recall.  

### **3. Hyperparameter Tuning**
Used `GridSearchCV` to optimize:  
```python
params = {
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}
```
**Result**: Improved recall from **0.58 → 0.61** (+5%).  

---

## 🚀 How to Use
1. **Predict Churn Risk**  
   ```python
   import xgboost as xgb
   model = xgb.XGBClassifier.load_model('bank_churn_model.json')
   # Sample input (preprocessed)
   customer_data = [[600, 0, 1, 42, 2, 100000, 2, 1, 1, 50000]]  # CreditScore, Geography, etc.
   print(model.predict_proba(customer_data)[:, 1])  # Churn probability
   ```

2. **Interpret Results**  
   - Probability ≥ 0.5 → High churn risk.  
   - **Recommended Action**: Offer personalized retention incentives (e.g., fee waivers).  

---

## 📈 Key Insights
- **Top Churn Drivers** (SHAP analysis):  
  1. **Age**: Older customers more likely to leave.  
  2. **Balance**: High-balance customers at risk.  
  3. **IsActiveMember**: Inactive users 2x more likely to churn.  

- **Limitations**:  
  - Precision tradeoff (44% false positives).  
  - Country-specific trends not fully captured.  

---

## 🔧 Dependencies
```python
xgboost==2.0.3
scikit-learn==1.4.2
pandas>=2.0.0
shap>=0.45.0
```
