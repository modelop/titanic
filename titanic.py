# modelop.schema.0: input_schema.avsc
# modelop.schema.1: output_schema.avsc

import pandas
import numpy
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import pickle
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


# modelop.init
def begin():

    global model
    model = pickle.load(open("RFC_model.pkl", "rb"))
    logger.info("'RFC_model.pkl' file loaded to global variable 'model'")

    global numeric_predictors, categorical_predictors, target_variable
    numeric_predictors = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_predictors = ["Sex", "Cabin", "Embarked"]
    target_variable = "Survived"
    logger.info("Variable roles assigned")


# modelop.score
def predict(scoring_df):

    # The smart-comment # modelop.recordsets.0: true encodes all input CSV data as a DataFrame
    logger.info("scoring_data is of shape: %s", scoring_df.shape)

    scoring_df["Prediction"] = model.predict(
        scoring_df[numeric_predictors + categorical_predictors]
    )
    # The smart-comment # modelop.recordsets.1: true yields a DataFrame as JSON-lines
    yield scoring_df.to_dict(orient="records")[0]


# modelop.metrics
def metrics(metrics_df):

    logger.info("metrics_df is of shape: %s", metrics_df.shape)

    X_test = metrics_df.drop("Survived", axis=1)
    y_true = metrics_df["Survived"]
    yield {
        "ACCURACY": model.score(
            X_test[numeric_predictors + categorical_predictors], y_true
        )
    }


# modelop.train
def train(training_df):

    logger.info("train_df is of shape: %s", training_df.shape)

    numeric_predictors = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_predictors = ["Sex", "Cabin", "Embarked"]
    target_variable = "Survived"

    training_df = training_df.loc[
        :, numeric_predictors + categorical_predictors + [target_variable]
    ]

    logger.info("Replacing Nulls")
    training_df.replace(to_replace=[None], value=numpy.nan, inplace=True)
    training_df[numeric_predictors] = training_df.loc[:, numeric_predictors].apply(
        pandas.to_numeric, errors="coerce"
    )

    logger.info("Setting 'y_train' and 'X_train'")
    X_train = training_df.drop("Survived", axis=1)
    y_train = training_df["Survived"]

    logger.info("Setting up numeric transformer Pipeline")
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    logger.info("Setting up categorical transformer Pipeline")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    logger.info("Initializing preprocessor")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_predictors),
            ("cat", categorical_transformer, categorical_predictors),
        ]
    )

    logger.info("Initializing model pipeline")
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    logger.info("Fitting model")
    model.fit(X_train, y_train)

    # pickle file should be written to outputDir/
    logger.info("Model fitting complete. Writing 'RFC_model.pkl' to outputDir/")
    with open("outputDir/RFC_model.pkl", "wb") as f:
        pickle.dump(model, f)

    logger.info("Training Job Complete!")
    pass