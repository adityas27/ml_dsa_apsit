# ML APSIT Sessions

This repository contains two student projects and a deployment directory:

- `housing/` and `housing_eda.ipynb` — exploratory analysis and exercises for the housing dataset.
- `chrun/` — churn analysis notebooks and modeling.
- `deployment/` — FastAPI app and artifacts for serving the trained churn model.
- `housing.csv` — housing dataset used in the housing notebook.

## Installation

There are two installation flows: one for the analysis (housing + churn notebooks) and a separate one for the deployment server.

1) Analysis (housing + churn)

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn jupyterlab notebook
```

This installs the packages students commonly need to run the notebooks in `housing/` and `chrun/`.

2) Deployment (FastAPI model server)

The production-ready dependencies are kept in `deployment/requirements.txt`.

```powershell
python -m venv .venv-deploy
.\.venv-deploy\Scripts\Activate.ps1
pip install -r deployment/requirements.txt
```

Run the server locally for development:

```powershell
uvicorn deployment.app:app --host 0.0.0.0 --port 8000 --reload
```

Ensure your trained model file is located at `deployment/model.pkl` before starting the server.

## Project structure

- `housing_eda.ipynb` — housing EDA notebook
- `housing.csv` — housing dataset used in notebooks
- `chrun/` — churn notebooks and modeling
- `deployment/` — FastAPI app, `requirements.txt`, and example request

## Example request (deployment)

See `deployment/example_request.json` for a sample payload for the `/predict` endpoint.

## Core Machine Learning Terms

🔹 Core Machine Learning Terms

1. Feature

An independent variable (input variable) used by a model to make predictions.

2. Target (Label)

The dependent variable the model is trying to predict.

3. Dataset

A structured collection of data consisting of features and (in supervised learning) a target variable.

4. Training Set

The portion of data used to train the model.

5. Test Set

The portion of data used to evaluate model performance on unseen data.

6. Train-Test Split

The process of dividing data into training and testing subsets to prevent overfitting and measure generalization.

7. Model

A mathematical function that maps input features to predictions.

8. Algorithm

The procedure or method used to train a model (e.g., Linear Regression, Random Forest).

9. Hyperparameter

A configuration parameter set before training that controls model behavior (e.g., learning rate, number of trees).

10. Parameter

Values learned by the model during training (e.g., weights in linear regression).

🔹 Model Performance & Behavior

11. Overfitting

When a model learns the training data too well, including noise, resulting in poor performance on unseen data.

12. Underfitting

When a model is too simple to capture patterns in data, leading to poor performance on both training and test data.

13. Bias

Error introduced due to overly simplistic assumptions in the model.

14. Variance

Error introduced due to sensitivity to small fluctuations in training data.

15. Bias-Variance Tradeoff

The balance between model simplicity (bias) and model complexity (variance).

16. Accuracy

The proportion of correct predictions out of total predictions.

17. Precision

The proportion of correctly predicted positive observations out of total predicted positives.

18. Recall

The proportion of correctly predicted positive observations out of all actual positives.

19. F1 Score

The harmonic mean of precision and recall.

20. Confusion Matrix

A table that shows true positives, true negatives, false positives, and false negatives.

🔹 EDA (Exploratory Data Analysis) Terms

21. Missing Values

Data points that are absent or undefined in a dataset.

22. Outlier

An observation that significantly deviates from other data points.

23. Distribution

The spread and frequency of values in a feature.

24. Skewness

Measure of asymmetry in data distribution.

25. Correlation

A statistical measure of how strongly two variables are related.

26. Multicollinearity

When two or more features are highly correlated with each other.

27. Data Imbalance

When one class significantly outnumbers another in classification tasks.

28. Data Leakage

When information from outside the training dataset is used to create the model, leading to overly optimistic performance.

🔹 Data Preprocessing Terms

29. Encoding

Converting categorical variables into numerical form.

30. One-Hot Encoding

A method of representing categorical variables as binary vectors.

31. Scaling

Normalizing feature values to a specific range (e.g., 0 to 1).

32. Standardization

Transforming data to have mean 0 and standard deviation 1.

33. Normalization

Rescaling values to a fixed range, typically [0,1].

## Datasets & Resources

- Housing dataset (local): `housing.csv`
- Churn notebooks & data (local): `chrun/`
- Telco Customer Churn (external, if you need the original dataset): https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Example housing datasets and competitions: https://www.kaggle.com/c/house-prices-advanced-regression-techniques


