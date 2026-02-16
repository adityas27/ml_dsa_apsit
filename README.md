# ML APSIT Sessions

A comprehensive machine learning project repository for learning and practicing data science concepts. This repository contains two complete student projects with exploratory data analysis, modeling, and a production-ready API for model deployment.

## 📁 Project Structure

### Housing Project
Learn regression modeling and housing price prediction:
- **[Housing EDA Notebook](https://github.com/adityas27/ml_apsit/blob/main/housing/housing_eda.ipynb)** — Exploratory data analysis of housing dataset
- **[Housing Models Notebook](https://github.com/adityas27/ml_apsit/blob/main/housing/housing_models.ipynb)** — Model training and evaluation
- **Data**: [`housing.csv`](https://github.com/adityas27/ml_apsit/blob/main/housing/housing.csv)

### Churn Prediction Project
Learn classification modeling and customer churn prediction:
- **[Churn EDA Notebook](https://github.com/adityas27/ml_apsit/blob/main/chrun/churn_eda.ipynb)** — Exploratory data analysis of customer churn
- **[Churn Models Notebook](https://github.com/adityas27/ml_apsit/blob/main/chrun/churn_model.ipynb)** — Model training and evaluation
- **Data**: [`churn_data.xlsx`](https://github.com/adityas27/ml_apsit/blob/main/chrun/churn_data.xlsx)
- **Trained Model**: [`logistic_churn_model.pkl`](https://github.com/adityas27/ml_apsit/blob/main/chrun/logistic_churn_model.pkl)

### Production Deployment
Ready-to-use FastAPI application for model serving:
- **[FastAPI App](https://github.com/adityas27/ml_apsit/blob/main/deployment/app.py)** — Production API server
- **[Requirements](https://github.com/adityas27/ml_apsit/blob/main/deployment/requirements.txt)** — Python dependencies
- **[Example Request](https://github.com/adityas27/ml_apsit/blob/main/deployment/example_request.json)** — Sample API payload
- **[Deployment README](https://github.com/adityas27/ml_apsit/blob/main/deployment/README.md)** — Detailed deployment guide

## 🚀 Quick Start Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/adityas27/ml_apsit.git
cd ml_apsit
```

### Step 2: Create a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

**For Notebooks (Housing + Churn Analysis):**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter jupyterlab
```

**For Deployment Server:**
```bash
pip install -r deployment/requirements.txt
```

## 📚 Running the Projects

### Running Jupyter Notebooks

After installing dependencies and activating the virtual environment:

```bash
jupyter notebook
```

Then navigate to:
- `housing/housing_eda.ipynb` — Start with exploratory analysis
- `housing/housing_models.ipynb` — Learn regression modeling
- `chrun/churn_eda.ipynb` — Customer churn data exploration
- `chrun/churn_model.ipynb` — Classification modeling

### Running the FastAPI Server

1. Ensure the trained model exists: `chrun/logistic_churn_model.pkl`
2. Activate the deployment environment:
   ```powershell
   .\.venv-deploy\Scripts\Activate.ps1
   ```
3. Start the server:
   ```bash
   uvicorn deployment.app:app --host 0.0.0.0 --port 8000 --reload
   ```
4. Test the API:
   - Visit: `http://localhost:8000/docs` (Interactive API documentation)
   - Use the example request: [`deployment/example_request.json`](https://github.com/adityas27/ml_apsit/blob/main/deployment/example_request.json)

## 📊 Key Concepts Learning Path

### Housing Regression Project
**Focus:** Predicting continuous values (house prices)

1. **[Start with EDA](https://github.com/adityas27/ml_apsit/blob/main/housing/housing_eda.ipynb)** - Understand data distribution, correlations, and patterns
2. **[Learn Modeling](https://github.com/adityas27/ml_apsit/blob/main/housing/housing_models.ipynb)** - Build and evaluate regression models
3. **Concepts covered:**
   - Data cleaning and preprocessing
   - Feature engineering and selection
   - Model comparison (Linear Regression, Random Forest, etc.)
   - Performance metrics (MAE, RMSE, R²)

### Churn Classification Project
**Focus:** Predicting categorical outcomes (churn/no churn)

1. **[Start with EDA](https://github.com/adityas27/ml_apsit/blob/main/chrun/churn_eda.ipynb)** - Analyze customer churn patterns
2. **[Learn Modeling](https://github.com/adityas27/ml_apsit/blob/main/chrun/churn_model.ipynb)** - Build classification models
3. **Concepts covered:**
   - Handling imbalanced data
   - Feature scaling and encoding
   - Model comparison (Logistic Regression, Decision Trees, etc.)
   - Classification metrics (Precision, Recall, F1-Score)
   - Deployment preparation

## 💾 Datasets

| Project | Dataset | Source | Format |
|---------|---------|--------|--------|
| Housing | [housing.csv](https://github.com/adityas27/ml_apsit/blob/main/housing/housing.csv) | Local | CSV (20,640 rows) |
| Churn | [churn_data.xlsx](https://github.com/adityas27/ml_apsit/blob/main/chrun/churn_data.xlsx) | Local | Excel | 
| Churn (Reference) | [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) | Kaggle | CSV |

## 📖 Machine Learning Terminology Reference

Use this section to understand key ML concepts you'll encounter in the notebooks.

### Core Concepts

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
### Model Performance & Evaluation

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
### Exploratory Data Analysis (EDA)

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
### Data Preprocessing Techniques

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

## 🌐 Useful Resources & References

### Documentation
- **Scikit-learn**: https://scikit-learn.org/stable/documentation.html
- **Pandas**: https://pandas.pydata.org/docs/
- **NumPy**: https://numpy.org/doc/
- **Matplotlib**: https://matplotlib.org/stable/contents.html
- **Seaborn**: https://seaborn.pydata.org/

### Learning Resources
- **Kaggle Datasets**: https://www.kaggle.com/datasets - Find more datasets to practice
- **Kaggle Learn**: https://www.kaggle.com/learn - Free micro-courses on ML topics
- **Scikit-learn Examples**: https://scikit-learn.org/stable/auto_examples/index.html
- **Real Python**: https://realpython.com/tutorials/basics/ - Python and data science tutorials

### Related Datasets
- **[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)** - Original churn dataset (Kaggle)
- **[House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)** - Advanced housing regression (Kaggle Competition)
- **[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)** - Free collection of datasets

### Tools & Libraries
- **Jupyter**: https://jupyter.org/ - Interactive notebook environment
- **JupyterLab**: https://jupyterlab.readthedocs.io/ - Next-gen notebook interface
- **FastAPI**: https://fastapi.tiangolo.com/ - Modern Python web framework for APIs
- **Uvicorn**: https://www.uvicorn.org/ - ASGI server for running FastAPI apps

## ❓ FAQs for Students

**Q: Which project should I start with?**
A: Start with the Housing project as it covers basic regression. Then move to Churn for classification concepts.

**Q: How long does it take to complete a project?**
A: ~2-3 hours per project if you study each notebook and run the code.

**Q: Can I modify the notebooks?**
A: Absolutely! Experimentation is key to learning. Try changing parameters, adding features, or using different models.

**Q: Where do I find more datasets to practice?**
A: Visit [Kaggle Datasets](https://www.kaggle.com/datasets) or [UCI ML Repository](https://archive.ics.uci.edu/ml/)

**Q: How do I deploy my own model?**
A: Use the deployment template in the `deployment/` directory as a reference for your own FastAPI setup.

## 📝 Project Workflow

```
1. Clone Repository
   ↓
2. Set up Virtual Environment
   ↓
3. Install Dependencies
   ↓
4. Run Housing Project
   ├─ Explore: housing_eda.ipynb
   └─ Model: housing_models.ipynb
   ↓
5. Run Churn Project
   ├─ Explore: churn_eda.ipynb
   └─ Model: churn_model.ipynb
   ↓
6. Understand Deployment (Optional)
   └─ Review: deployment/app.py & README.md
```

## 📧 Questions or Issues?

For questions about this repository, please open an issue on GitHub: https://github.com/adityas27/ml_apsit/issues

---

**Happy Learning! 🚀**

*Last Updated: February 2025*


