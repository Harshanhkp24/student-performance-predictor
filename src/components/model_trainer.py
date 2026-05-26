import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import evaluate_models, save_json, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    model_report_file_path=os.path.join("artifacts","model_report.json")
    model_summary_file_path=os.path.join("artifacts","model_summary.json")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(max_iter=20000),
                "ElasticNet": ElasticNet(max_iter=20000),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
                "XGBRegressor": XGBRegressor(
                    random_state=42,
                    objective="reg:squarederror",
                    verbosity=0,
                ),
                "CatBoosting Regressor": CatBoostRegressor(
                    verbose=False,
                    random_seed=42,
                ),
            }
            params={
                "Linear Regression": {},
                "Ridge": {
                    "alpha": [0.1, 1.0, 10.0, 25.0, 50.0, 100.0]
                },
                "Lasso": {
                    "alpha": [0.0005, 0.001, 0.01, 0.1, 1.0]
                },
                "ElasticNet": {
                    "alpha": [0.0005, 0.001, 0.01, 0.1],
                    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                },
                "KNeighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2],
                },
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                    ],
                    "max_depth": [None, 4, 6, 8],
                    "min_samples_leaf": [1, 2, 4],
                },
                "Random Forest":{
                    "n_estimators": [128, 256],
                    "max_depth": [None, 8, 12],
                    "min_samples_leaf": [1, 2, 4],
                },
                "Gradient Boosting":{
                    "learning_rate": [0.03, 0.05, 0.1],
                    "subsample": [0.7, 0.8, 0.9, 1.0],
                    "n_estimators": [64, 128, 256],
                    "max_depth": [2, 3],
                },                
                "XGBRegressor":{
                    "learning_rate": [0.03, 0.05, 0.1],
                    "n_estimators": [64, 128, 256],
                    "max_depth": [3, 4, 5],
                    "subsample": [0.8, 1.0],
                },
                "CatBoosting Regressor":{
                    "depth": [4, 6, 8],
                    "learning_rate": [0.03, 0.05, 0.1],
                    "iterations": [100, 200, 300],
                },
                "AdaBoost Regressor":{
                    "learning_rate": [0.01, 0.05, 0.1, 0.5],
                    "n_estimators": [64, 128, 256],
                }
                
            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )
            
            sorted_models = sorted(
                model_report.items(),
                key=lambda item: item[1]["test_r2"],
                reverse=True,
            )
            best_model_name, best_model_metrics = sorted_models[0]
            best_model_score = best_model_metrics["test_r2"]
            best_model = best_model_metrics["best_estimator"]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            serializable_report = []
            for model_name, metrics in sorted_models:
                serializable_report.append(
                    {
                        "model_name": model_name,
                        "train_r2": metrics["train_r2"],
                        "test_r2": metrics["test_r2"],
                        "test_mae": metrics["test_mae"],
                        "best_params": metrics["best_params"],
                    }
                )

            model_summary = {
                "best_model_name": best_model_name,
                "best_model_test_r2": best_model_metrics["test_r2"],
                "best_model_train_r2": best_model_metrics["train_r2"],
                "best_model_test_mae": best_model_metrics["test_mae"],
                "best_model_params": best_model_metrics["best_params"],
                "engineered_features": [
                    "language_average",
                    "language_total",
                    "language_gap",
                ],
            }

            save_json(
                file_path=self.model_trainer_config.model_report_file_path,
                payload=serializable_report,
            )
            save_json(
                file_path=self.model_trainer_config.model_summary_file_path,
                payload=model_summary,
            )

            return best_model_score
            



            
        except Exception as e:
            raise CustomException(e,sys)
