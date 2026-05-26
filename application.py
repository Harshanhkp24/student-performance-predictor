import json
import os

from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)
app = application

DEFAULT_MODEL_SUMMARY = {
    "best_model_name": "ElasticNet",
    "best_model_test_r2": 0.8808,
    "best_model_test_mae": 4.2075,
    "engineered_features": [
        "language_average",
        "language_total",
        "language_gap",
    ],
}


def load_model_summary():
    summary_path = os.path.join("artifacts", "model_summary.json")
    if not os.path.exists(summary_path):
        return DEFAULT_MODEL_SUMMARY

    with open(summary_path, "r", encoding="utf-8") as summary_file:
        return json.load(summary_file)


MODEL_SUMMARY = load_model_summary()
BEST_MODEL_NAME = MODEL_SUMMARY["best_model_name"]
BEST_MODEL_R2 = f'{MODEL_SUMMARY["best_model_test_r2"]:.4f}'
BEST_MODEL_MAE = f'{MODEL_SUMMARY["best_model_test_mae"]:.2f}'

PROJECT_STATS = [
    {"value": "1,000", "label": "Student records analyzed"},
    {"value": "7 + 3 engineered", "label": "Raw and derived input features"},
    {"value": "800 / 200", "label": "Train and test split"},
    {"value": BEST_MODEL_NAME, "label": "Best saved model"},
    {"value": BEST_MODEL_R2, "label": "R2 score on test set"},
    {"value": BEST_MODEL_MAE, "label": "Mean absolute error"},
]

WORKFLOW_STEPS = [
    {
        "title": "Business Understanding",
        "description": (
            "Framed the problem as a regression task to estimate math score from"
            " demographic context, lunch support, test preparation, and language"
            " subject performance."
        ),
    },
    {
        "title": "EDA in Notebook",
        "description": (
            "Validated data quality, reviewed distributions, compared category"
            " behavior, and studied relationships between reading, writing, and"
            " math scores."
        ),
    },
    {
        "title": "Data Pipeline",
        "description": (
            "Built reusable ingestion and preprocessing components using"
            " train-test split, imputers, one-hot encoding, feature scaling,"
            " and EDA-driven score features such as language average and gap."
        ),
    },
    {
        "title": "Model Benchmarking",
        "description": (
            "Compared linear, regularized linear, distance-based, bagging,"
            " boosting, XGBoost, and CatBoost regressors with GridSearchCV"
            " before saving the strongest artifact."
        ),
    },
    {
        "title": "Inference API",
        "description": (
            "Wrapped the model and preprocessor in a prediction pipeline that"
            " transforms a single student profile into a math score estimate."
        ),
    },
    {
        "title": "Web Deployment",
        "description": (
            "Exposed the predictor through Flask templates so the project can be"
            " shared as a live portfolio app with a polished interface."
        ),
    },
]

EDA_INSIGHTS = [
    {
        "title": "Reading and writing are strong signals",
        "description": (
            "Pairplots and score comparisons show a clear linear relationship"
            " across the three subjects, making reading and writing useful"
            " predictors for math performance."
        ),
    },
    {
        "title": "Lunch support matters",
        "description": (
            "Students with standard lunch generally perform better than students"
            " with free or reduced lunch, which suggests a socioeconomic effect"
            " in the dataset."
        ),
    },
    {
        "title": "Female students lead overall averages",
        "description": (
            "The EDA notes that female students tend to score better overall,"
            " while male students show stronger math performance on average."
        ),
    },
    {
        "title": "Preparation course improves outcomes",
        "description": (
            "Students who completed the test preparation course tend to score"
            " better than those who did not complete it."
        ),
    },
]

MODEL_CANDIDATES = [
    "Linear Regression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "KNeighbors Regressor",
    "Random Forest Regressor",
    "Decision Tree Regressor",
    "Gradient Boosting Regressor",
    "XGBRegressor",
    "CatBoost Regressor",
    "AdaBoost Regressor",
]

FEATURE_GROUPS = {
    "categorical": [
        "gender",
        "race_ethnicity",
        "parental_level_of_education",
        "lunch",
        "test_preparation_course",
    ],
    "numerical": [
        "reading_score",
        "writing_score",
        *MODEL_SUMMARY["engineered_features"],
    ],
}

MODEL_METRICS = {
    "best_model": BEST_MODEL_NAME,
    "r2_score": BEST_MODEL_R2,
    "mae": BEST_MODEL_MAE,
    "target": "math_score",
}

SAMPLE_PREDICTIONS = [
    {
        "title": "Prepared high performer",
        "summary": (
            "Female student, group C, bachelor's degree at home, standard lunch,"
            " completed preparation course."
        ),
        "reading_score": 88,
        "writing_score": 90,
        "prediction": "77.26",
    },
    {
        "title": "Steady classroom performer",
        "summary": (
            "Male student, group D, some college background, standard lunch, no"
            " preparation course."
        ),
        "reading_score": 72,
        "writing_score": 70,
        "prediction": "76.99",
    },
    {
        "title": "Needs support profile",
        "summary": (
            "Female student, group B, high school parental education,"
            " free/reduced lunch, no preparation course."
        ),
        "reading_score": 55,
        "writing_score": 58,
        "prediction": "48.56",
    },
]


def build_template_context():
    return {
        "project_stats": PROJECT_STATS,
        "workflow_steps": WORKFLOW_STEPS,
        "eda_insights": EDA_INSIGHTS,
        "model_candidates": MODEL_CANDIDATES,
        "feature_groups": FEATURE_GROUPS,
        "model_metrics": MODEL_METRICS,
        "sample_predictions": SAMPLE_PREDICTIONS,
        "form_values": {},
    }


def get_performance_band(prediction):
    if prediction >= 80:
        return (
            "High performance band",
            "The model expects a strong math outcome with the submitted profile.",
        )
    if prediction >= 60:
        return (
            "Developing well",
            "The profile suggests a solid performance range with room to improve.",
        )
    return (
        "Needs focused support",
        "The profile suggests the student may benefit from additional math support.",
    )


@app.route("/")
def index():
    return render_template("index.html", **build_template_context())


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    context = build_template_context()

    if request.method == "GET":
        return render_template("home.html", **context)

    form_values = request.form.to_dict()
    data = CustomData(
        gender=form_values.get("gender"),
        race_ethnicity=form_values.get("race_ethnicity"),
        parental_level_of_education=form_values.get(
            "parental_level_of_education"
        ),
        lunch=form_values.get("lunch"),
        test_preparation_course=form_values.get("test_preparation_course"),
        reading_score=float(form_values.get("reading_score")),
        writing_score=float(form_values.get("writing_score")),
    )

    pred_df = data.get_data_as_data_frame()
    predict_pipeline = PredictPipeline()
    prediction = round(float(predict_pipeline.predict(pred_df)[0]), 2)
    performance_band, performance_note = get_performance_band(prediction)

    context.update(
        {
            "results": prediction,
            "performance_band": performance_band,
            "performance_note": performance_note,
            "form_values": form_values,
        }
    )
    return render_template("home.html", **context)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
