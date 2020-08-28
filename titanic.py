import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import pickle


#modelop.init
def begin():
    global model
    model = pickle.load(open('model.pkl', 'rb'))



#modelop.train
def train(train_df):
    X_train = train_df.drop('Survived', axis=1)
    y_train = train_df['Survived']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns

    categorical_features = X_train.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier())])

    model.fit(X_train, y_train)
    
    with open('outputDir/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    pass

#modelop.metrics
def metrics(df):

    X_test = df.drop('Survived', axis=1)
    y_test = df['Survived']
    yield { "ACCURACY": model.score(X_test, y_test)}

#modelop.score
def predict(X):
    df = pd.DataFrame(X, index=[0])
    y_pred = model.predict(df)
    for p in y_pred:
        yield p










