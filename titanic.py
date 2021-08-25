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


# modelop.init
def begin():

    global model
    model = pickle.load(open("model.pkl", "rb"))

    logger.info("'model.pkl' file loaded to global variable 'model'")


# modelop.score
def predict(X):
    df = pandas.DataFrame(X, index=[0])
    y_pred = model.predict(df)
    for p in y_pred:
        yield p


# modelop.metrics
def metrics(df):

    X_test = df.drop("Survived", axis=1)
    y_test = df["Survived"]
    yield {"ACCURACY": model.score(X_test, y_test)}


# modelop.train
def train(train_df):

    # Turn input data into a DataFrame
    train_df = pandas.DataFrame(train_df, index=[0])

    numeric_columns = [
        "PassengerId",
        "Survived",
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
    ]

    logger.info("Replacing Nulls")
    train_df.replace(to_replace=[None], value=numpy.nan, inplace=True)
    train_df[numeric_columns] = train_df.loc[:, numeric_columns].apply(
        pandas.to_numeric, errors="coerce"
    )

    logger.info("Setting 'y_train' and 'X_train'")
    X_train = train_df.drop("Survived", axis=1)
    y_train = train_df["Survived"]

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

    logger.info("Selecting numeric and categorical features by dtype")
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X_train.select_dtypes(include=["object"]).columns

    logger.info("Initializing preprocessor")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    logger.info("Initializing model pipeline")
    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier())]
    )

    logger.info("Fitting model")
    model.fit(X_train, y_train)

    # pickle file should be written to outputDir
    logger.info("Model fitting complete. Writing model.pkl to outputDir")
    with open("outputDir/model.pkl", "wb") as f:
        pickle.dump(model, f)

    logger.info("Training Job Complete!")
    pass


# For local testing
if __name__ == "__main__":
    train_df = pandas.read_csv("train.csv")
    test_df = pandas.read_csv("test.csv")
    pred_df = pandas.read_csv("predict.csv")

    train(train_df)
    begin()

    X = [[519, 2, "Bob", "male", 36.0, 1, 0, 226875, 26.0, "C26", "S"]]
    print(predict(X))

    for m in metrics(test_df):
        print(m)
