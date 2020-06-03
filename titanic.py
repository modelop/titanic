import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import pickle



def begin():
    global model
    model = pickle.load(open('model.pkl', 'rb'))



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

    #where do we write this?? s3?  fixed location on contaier?  or get location from env variable?
    pickle.dump(model, open('model.pkl', 'wb'))


def metrics(test):
    X_test = test.drop('Survived', axis=1)
    y_test = test['Survived']
    yield { "ACCURACY": model.score(X_test, y_test)}


def action(datum):
    df = pd.DataFrame(datum, index=[0])
    y_pred = model.predict(df)
    yield y_pred  #yield one column at a time...




if __name__ == "__main__":
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    pred_df = pd.read_csv('predict.csv')

    train(train_df)
    begin()
    for m in metrics(test_df):
        print(m)

    for a in action(pred_df):
        print(a)






