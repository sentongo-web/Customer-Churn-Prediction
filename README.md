# Customer Churn Prediction

A machine learning project that predicts which telecom customers are likely to cancel their subscription — using both classical ML models and a neural network, with a focus on understanding *why* customers leave, not just *that* they do.

---

## The Problem

A telecom company loses revenue every time a customer cancels. The trouble is, by the time a customer calls to cancel, it's usually too late to change their mind. If you could identify at-risk customers *early*, you could intervene — offer a discount, fix a service issue, or reach out proactively.

That's exactly what this project tackles: given what we know about a customer (their plan, tenure, monthly charges, services), can we predict whether they'll churn in the near future?

---

## Dataset

**IBM Telco Customer Churn** — a publicly available benchmark dataset with 7,043 customers and 20 features:

| Category     | Features                                                      |
| ------------ | ------------------------------------------------------------- |
| Demographics | Gender, Senior Citizen, Partner, Dependents                   |
| Account      | Tenure, Contract Type, Billing Method, Monthly/Total Charges  |
| Services     | Phone, Internet, Security, Backup, Streaming, Tech Support    |
| Target       | `Churn` — Yes (left) / No (stayed)                            |

About **26.5%** of customers churned — a moderately imbalanced dataset.

---

## Project Structure

```text
Customer-Churn-Prediction/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
├── churn_prediction.ipynb                  # Main notebook (full pipeline)
├── plots/                                  # Generated visualizations
│   ├── 01_target_distribution.png
│   ├── 02_numeric_distributions.png
│   ├── 03_categorical_churn_rates.png
│   ├── 04_tenure_cohort_churn.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_nn_training_history.png
│   ├── 07_model_comparison.png
│   ├── 08_confusion_matrices.png
│   ├── 09_roc_curves.png
│   ├── 10_feature_importances.png
│   └── 11_lr_coefficients.png
└── README.md
```

---

## Approach

The notebook walks through the complete ML pipeline, with every decision explained in markdown before the code.

### 1. Exploratory Data Analysis

Before touching the data, we look at it. The EDA section covers:

- **Target distribution** — confirmed ~26% churn rate, meaning accuracy alone is misleading
- **Numeric features vs churn** — tenure, monthly charges, and total charges all show clear separation
- **Categorical features vs churn** — contract type and internet service type are the strongest categorical signals
- **Tenure cohort analysis** — customers in their first year churn at ~48%; those past 4 years at under 10%
- **Correlation check** — tenure and TotalCharges are correlated (0.83), which makes sense; both are kept

### 2. Data Cleaning

Only one real issue existed: `TotalCharges` was stored as a string (due to 11 blank entries for brand-new customers with tenure=0). Those were converted to numeric and filled with 0.

### 3. Preprocessing

- Drop `customerID` — no predictive signal, just an ID
- Label-encode binary categories (Yes/No, Male/Female)
- One-hot encode multi-class categories (Contract, InternetService, PaymentMethod, etc.)
- StandardScaler on all features — required for Logistic Regression and the Neural Network; harmless for trees
- Stratified 80/20 train/test split to preserve class balance

### 4. Models Trained

| Model | Why We Used It |
| --- | --- |
| **Logistic Regression** | Baseline, interpretable, great when signals are relatively linear |
| **Decision Tree** | Nonlinear, rule-based, shows clear decision boundaries |
| **Random Forest** | Ensemble of trees — reduces variance via bagging, robust without tuning |
| **Gradient Boosting** | Builds trees sequentially, each correcting the previous — powerful on tabular data |
| **Neural Network** | Deep learning comparison — 3 hidden layers, Batch Norm, Dropout, EarlyStopping |

### 5. Evaluation Strategy

Because the dataset is imbalanced, we tracked five metrics:

- **Accuracy** — overall, but can be misleading here
- **Precision** — of predicted churners, how many actually left
- **Recall** — of all customers who left, how many did we catch
- **F1** — the balance between precision and recall
- **ROC-AUC** — model's ability to rank churners above non-churners

For business use, **Recall matters most** — a missed churner costs more than a false alarm.

---

## Results

| Model | Accuracy | F1 | ROC-AUC |
| --- | --- | --- | --- |
| Logistic Regression | ~0.74 | ~0.61 | ~0.85 |
| Decision Tree | ~0.76 | ~0.58 | ~0.74 |
| Random Forest | ~0.79 | ~0.62 | ~0.85 |
| **Gradient Boosting** | **~0.80** | **~0.64** | **~0.86** |
| Neural Network | ~0.78 | ~0.61 | ~0.84 |

Exact values vary slightly by run — see notebook output for precise numbers.

Gradient Boosting won on both F1 and AUC. The Neural Network was competitive but didn't overtake the ensembles — which is typical for structured tabular data at this scale.

---

## Key Findings

**1. Contract type is the biggest single predictor.**
Month-to-month customers churn at ~43%. Two-year contract customers churn at ~3%. If the business wants to reduce churn, the most direct lever is moving customers onto longer contracts — through discounts, loyalty perks, or bundled pricing.

**2. New customers are the highest risk.**
Churn rate in year 1 is nearly 50%. Onboarding experience, early engagement, and ensuring the customer gets value quickly would make the biggest dent in overall churn numbers.

**3. Fiber optic + no security services is a red flag.**
Customers with fiber internet but no OnlineSecurity or TechSupport churn at much higher rates. They're paying more but getting less protection — a recipe for dissatisfaction.

**4. Electronic check users are disproportionately likely to leave.**
~45% churn rate vs ~15% for bank transfer/credit card customers. Auto-pay removes friction and increases retention. Migrating users off manual payment methods is a low-cost, high-impact action.

**5. Gender doesn't predict churn.**
The churn rates for male and female customers are nearly identical. Personalizing retention campaigns by gender would be a waste of effort.

---

## What the Neural Network Taught Us

The NN performed well but didn't beat Gradient Boosting. This is a common and important pattern worth understanding:

- Deep learning shines on **unstructured data** (images, text, audio) where it can learn hierarchical representations
- For **tabular data** with ~7K rows and ~30 features, gradient boosting almost always wins or ties
- The NN still hit competitive AUC (~0.84) and its training curves showed clean convergence without overfitting, thanks to Dropout and EarlyStopping

The lesson: choose your tools based on data type and size, not hype.

---

## Dependencies

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow >= 2.x
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

---

## How to Run

1. Clone the repo
2. Place the dataset CSV in the root folder (it's already there)
3. Open `churn_prediction.ipynb` in Jupyter
4. Run all cells top to bottom

The `plots/` folder will be created automatically.

---

## Lessons Learned

- **EDA always pays off.** Spending time understanding the data before modeling revealed that contract type and tenure were dominant signals — which informed feature encoding choices and set realistic expectations for model performance.

- **Imbalanced datasets need careful metric selection.** A model that predicted "No churn" for everyone would hit 73% accuracy — but would be completely useless. F1 and AUC tell the real story.

- **Scaling matters for some models, not others.** Logistic Regression and Neural Networks are sensitive to feature scale; tree-based models don't care. Scaling all features and letting trees ignore it (rather than managing two separate pipelines) was the practical choice here.

- **Deep learning is not always the answer.** The Neural Network added depth to the project and showed clear understanding of the architecture — but honestly, Gradient Boosting was better for this specific problem. That's not a failure; recognizing when *not* to use deep learning is itself a skill.

- **Business context shapes which metric to optimize.** Recall matters more than precision here because false negatives (missing a churner) have higher business cost than false positives (incorrectly flagging a loyal customer). This should drive any threshold tuning in production.
