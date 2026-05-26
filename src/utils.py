import json
import os
import pickle
import sys

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def save_json(file_path, payload):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            para = param.get(model_name, {})

            gs = GridSearchCV(model, para, cv=5, scoring="r2", n_jobs=1)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            report[model_name] = {
                "best_estimator": best_model,
                "best_params": gs.best_params_,
                "train_r2": float(train_model_score),
                "test_r2": float(test_model_score),
                "test_mae": float(test_mae),
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
