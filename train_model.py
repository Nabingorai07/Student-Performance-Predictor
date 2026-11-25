"""Retrain the Enhanced Linear Regression pipeline and save it as a .joblib file.

Run this if the app says "Could not load the trained model" because the
scikit-learn installed on your machine differs from the version that created
the shipped model file (1.6.1).

    python train_model.py

It reproduces notebooks/03-linear-regression.ipynb (enhanced model, same
split and settings) using the scikit-learn installed in YOUR environment,
so the saved file will always load there.
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))

NUMERIC = [
    "Age", "AttendancePct", "StudyHoursPerWeek", "PreviousGrade",
    "AssignmentMarks", "InternalAssessmentMarks",
    "ExtracurricularActivities", "AttendanceStudyInteraction",
]
CATEGORICAL = ["Gender", "OnlineClasses", "ParentalSupport"]
FEATURES = [
    "Age", "Gender", "AttendancePct", "StudyHoursPerWeek", "PreviousGrade",
    "AssignmentMarks", "InternalAssessmentMarks", "OnlineClasses",
    "ExtracurricularActivities", "ParentalSupport",
    "AttendanceStudyInteraction",
]
TARGET = "FinalExamMarks"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.join(
            HERE, "data", "preprocessed",
            "student_performance_feature_engineered_v2.csv"),
    )
    parser.add_argument(
        "--out", default=os.path.join(HERE, "linear_regression_final.joblib")
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUMERIC),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL),
        ])),
        ("regressor", LinearRegression()),
    ])
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    print(f"MAE  : {mean_absolute_error(y_te, pred):.4f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_te, pred)):.4f}")
    print(f"R2   : {r2_score(y_te, pred):.4f}")

    joblib.dump(model, args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
