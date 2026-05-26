# Student Performance Indicator

An end-to-end machine learning project that predicts a student's **math score** from demographic, academic, and support-related features. The project starts with exploratory data analysis in Jupyter notebooks, moves through a reusable training pipeline, and ends with a Flask web application that serves live predictions.

## Project Overview

This repository showcases the full lifecycle of a supervised machine learning project:

- Problem understanding and dataset familiarization
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Model training and algorithm comparison
- Artifact saving for reuse in inference
- Flask-based frontend for live prediction
- Vercel-ready deployment structure for GitHub showcase

The project is built around the Student Performance dataset and focuses on estimating `math_score` from:

- `gender`
- `race_ethnicity`
- `parental_level_of_education`
- `lunch`
- `test_preparation_course`
- `reading_score`
- `writing_score`

## What Have we Done

### 1. Performed EDA in notebook

EDA was completed in:

- `notebook/1 . EDA STUDENT PERFORMANCE .ipynb`

In this notebook, you:

- Imported the dataset and inspected its structure
- Verified the dataset shape: **1000 rows and 8 columns**
- Checked missing values and found **no missing data**
- Checked duplicate rows and found **no duplicates**
- Reviewed data types and unique values
- Analyzed descriptive statistics for the score columns
- Added total and average score views for deeper understanding
- Created univariate, bivariate, and multivariate visualizations
- Compared performance across gender, ethnicity, lunch type, parental education, and test preparation
- Used pairplots and score relationships to understand correlation between subject scores

### 2. Identified key EDA insights

Important insights recorded in the notebook include:

- Reading, writing, and math scores move in a strongly linear way
- Students with **standard lunch** tend to perform better than students with **free/reduced lunch**
- Female students show stronger overall performance on average
- Male students tend to score better in math on average
- Students who completed the **test preparation course** generally score better
- Group E students tend to perform best, while Group A students tend to perform lower
- Reading and writing scores are very useful predictors for math score

### 3. Built a modular ML pipeline

The codebase is organized into reusable components under `src/`:

- `src/components/data_ingestion.py`
  - Reads the CSV dataset
  - Saves raw data to `artifacts/data.csv`
  - Splits data into train and test sets
  - Saves `artifacts/train.csv` and `artifacts/test.csv`

- `src/components/data_transformation.py`
  - Separates categorical and numerical features
  - Uses `ColumnTransformer` for preprocessing
  - Numerical pipeline:
    - `SimpleImputer(strategy="median")`
    - `StandardScaler()`
  - Categorical pipeline:
    - `SimpleImputer(strategy="most_frequent")`
    - `OneHotEncoder()`
    - `StandardScaler(with_mean=False)`
  - Saves the fitted preprocessor to `artifacts/preprocessor.pkl`

- `src/components/model_trainer.py`
  - Trains and compares multiple regression models
  - Tunes models using `GridSearchCV`
  - Saves the best model to `artifacts/model.pkl`

- `src/pipeline/predict_pipeline.py`
  - Loads the saved model and preprocessor
  - Applies preprocessing to incoming data
  - Returns a math score prediction

### 4. Trained multiple regression models

The training pipeline compares the following algorithms:

- Linear Regression
- Random Forest Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor
- XGBRegressor
- CatBoost Regressor
- AdaBoost Regressor

## Current Saved Model Performance

The currently saved artifact in `artifacts/model.pkl` is:

- **Best saved model:** `LinearRegression`
- **R2 score on `artifacts/test.csv`:** `0.8778`
- **Mean Absolute Error:** `4.2425`

This means the model explains a strong portion of the variation in math scores and is usually off by about **4.24 points** on average on the saved test split.

## Sample Model Predictions

### A. Example predictions from the saved model

These are sample profiles passed through the saved artifact:

| Scenario | Reading Score | Writing Score | Predicted Math Score |
| --- | ---: | ---: | ---: |
| Prepared high performer | 88 | 90 | 78.00 |
| Steady classroom performer | 72 | 70 | 76.81 |
| Needs support profile | 55 | 58 | 48.69 |

### B. Sample held-out test examples

These examples come from `artifacts/test.csv` and compare actual vs predicted values:

| Profile Summary | Reading | Writing | Actual Math | Predicted Math | Abs Error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Female, Group D, associate's degree, standard lunch, no prep | 95 | 89 | 82.00 | 82.00 | 0.00 |
| Male, Group E, some college, standard lunch, no prep | 57 | 52 | 66.00 | 65.69 | 0.31 |
| Female, Group B, associate's degree, free/reduced lunch, no prep | 61 | 55 | 46.00 | 46.62 | 0.62 |

## Tech Stack And Libraries Used

- Python
- Flask
- pandas
- NumPy
- scikit-learn
- XGBoost
- CatBoost
- dill / pickle for artifact persistence
- Matplotlib
- Seaborn
- Jupyter Notebook

## Project Structure

```text
mlproject/
|-- application.py
|-- app.py
|-- requirements.txt
|-- README.md
|-- artifacts/
|   |-- data.csv
|   |-- train.csv
|   |-- test.csv
|   |-- preprocessor.pkl
|   |-- model.pkl
|-- notebook/
|   |-- 1 . EDA STUDENT PERFORMANCE .ipynb
|   |-- 2. MODEL TRAINING.ipynb
|   |-- data/
|       |-- stud.csv
|-- src/
|   |-- components/
|   |   |-- data_ingestion.py
|   |   |-- data_transformation.py
|   |   |-- model_trainer.py
|   |-- pipeline/
|   |   |-- predict_pipeline.py
|   |-- exception.py
|   |-- logger.py
|   |-- utils.py
|-- templates/
|   |-- index.html
|   |-- home.html
```

## Frontend And Demo Experience

The frontend now includes:

- A full landing page describing the project story
- EDA highlights and workflow explanation
- Saved model metrics
- Sample prediction cards
- A polished prediction form UI
- A prediction result panel with performance band messaging

This makes the repository much stronger as a **GitHub portfolio project** rather than only a training notebook or a basic HTML form.

## How The Prediction App Works

### Route: `/`

The landing page explains:

- What the project does
- What steps were followed
- Which models were compared
- What insights came from EDA
- What performance the current model achieved

### Route: `/predictdata`

The prediction page:

1. Takes user input from the form
2. Builds a single-row dataframe using `CustomData`
3. Loads `artifacts/preprocessor.pkl`
4. Loads `artifacts/model.pkl`
5. Applies preprocessing
6. Predicts the `math_score`
7. Displays the result in the browser

## Run The Project Locally

### 1. Create and activate your environment

On Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the Flask app

```powershell
python application.py
```

Then open:

```text
http://127.0.0.1:5000/
```

## Re-Train The Pipeline

To regenerate the train/test splits, preprocessor, and model artifact:

```powershell
python src/components/data_ingestion.py
```

That script:

- Reads `notebook/data/stud.csv`
- Splits the data
- Applies preprocessing
- Trains the selected models
- Saves the updated artifacts

## Deploy On Vercel

This repo is now structured to be Vercel-friendly for Flask deployment.

### What was updated for deployment

- Added a root `app.py` entrypoint for Vercel
- Kept the Flask `app` instance importable
- Preserved template rendering from the repository root
- Kept saved artifacts inside the repo so the deployed app can run inference immediately

### Deployment steps

#### Option 1: Deploy from GitHub

1. Push this repository to GitHub
2. Go to Vercel and import the GitHub repository
3. Let Vercel detect the project automatically
4. Deploy

#### Option 2: Deploy from the Vercel CLI

```powershell
npm i -g vercel
vercel
```

For production deployment:

```powershell
vercel --prod
```

### Important deployment note

The live predictor depends on:

- `artifacts/model.pkl`
- `artifacts/preprocessor.pkl`

If these files are missing from the deployed repository, prediction will fail. Keep them committed if you want the demo to work immediately after deployment.

## Improvements Made While Polishing This Project

During this update, the project was improved by:

- Fixing the swapped reading and writing score mapping in the Flask form flow
- Aligning the training and inference preprocessor filename to `preprocessor.pkl`
- Removing inference debug prints from the prediction pipeline
- Making dataset path loading more portable across environments
- Improving the frontend from basic HTML into a complete ML showcase
- Adding a Vercel-compatible app entrypoint
- Writing a portfolio-quality README

## Future Improvements

- Add model evaluation plots and screenshots to the README
- Add unit tests for preprocessing and prediction routes
- Add Docker support
- Add CI checks for linting and smoke tests
- Show feature importance or coefficient interpretation
- Track experiments with MLflow or Weights & Biases

## Final Summary

You have already completed the core ML work:

- EDA on student performance data
- Data preprocessing
- Model training and comparison
- Artifact generation
- Prediction pipeline creation

This update turns that work into a **complete, presentable ML project** with:

- Better documentation
- Better frontend
- Better inference flow
- GitHub-ready structure
- Vercel-ready deployment path
