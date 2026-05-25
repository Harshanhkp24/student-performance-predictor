from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)
app = application

PROJECT_STATS = [
    {"value": "1,000", "label": "Student records analyzed"},
    {"value": "7", "label": "Input features used"},
    {"value": "800 / 200", "label": "Train and test split"},
    {"value": "LinearRegression", "label": "Best saved model"},
    {"value": "0.8778", "label": "R2 score on test set"},
    {"value": "4.24", "label": "Mean absolute error"},
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
            " train-test split, imputers, one-hot encoding, and feature scaling."
        ),
    },
    {
        "title": "Model Benchmarking",
        "description": (
            "Compared multiple regression algorithms with GridSearchCV and saved"
            " the strongest trained artifact for inference."
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
            " shared as a live portfolio app and deployed on Vercel."
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
    "numerical": ["reading_score", "writing_score"],
}

MODEL_METRICS = {
    "best_model": "LinearRegression",
    "r2_score": "0.8778",
    "mae": "4.24",
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
        "prediction": "78.00",
    },
    {
        "title": "Steady classroom performer",
        "summary": (
            "Male student, group D, some college background, standard lunch, no"
            " preparation course."
        ),
        "reading_score": 72,
        "writing_score": 70,
        "prediction": "76.81",
    },
    {
        "title": "Needs support profile",
        "summary": (
            "Female student, group B, high school parental education,"
            " free/reduced lunch, no preparation course."
        ),
        "reading_score": 55,
        "writing_score": 58,
        "prediction": "48.69",
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
