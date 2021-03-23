
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import pickle


# modelop.init
def begin():
    
    print("Scope: init function", flush=True)

    global model
    model = pickle.load(open('model.pkl', 'rb'))
    
    print("pkl file loaded to global variable", flush=True)


# modelop.train
def train(train_df):
    
    print("Scope: training function", flush=True)
    
    print("input_type: ", type(train_df), flush=True)
    
    train_df = pd.DataFrame(train_df, index=[0])
    
    numeric_columns = [
        'PassengerId', 'Survived', 'Pclass','Age', 'SibSp', 
        'Parch', 'Fare'
    ]
    
    print("Replacing nulls", flush=True)
    
    train_df.replace(to_replace=[None], value=np.nan, inplace=True)
    train_df[numeric_columns] = train_df.loc[:, numeric_columns] \
            .apply(pd.to_numeric, errors='coerce')
    
    print("Setting y_train and X_train", flush=True)
    
    X_train = train_df.drop('Survived', axis=1)
    y_train = train_df['Survived']

    print("Setting up numeric transformer Pipeline", flush=True)
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    print("Setting up categorical transformer Pipeline", flush=True)

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    print("Selecting numeric and categorical features by dtype", flush=True)
    
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns

    categorical_features = X_train.select_dtypes(include=['object']).columns

    print("Initializing preprocessor", flush=True)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])


    print("Initializing model pipeline", flush=True)
    
    model = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier())])

    print("Fitting model", flush=True)
    
    model.fit(X_train, y_train)
    
    print("model fitting complete. Writing .pkl to outputDir", flush=True)
    
    #where do we write this?? s3?  fixed location on contaier?  or get location from env variable?
    with open('outputDir/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Finished Training. Now exiting train function", flush=True)
    pass

# modelop.metrics
def metrics(df):

    X_test = df.drop('Survived', axis=1)
    y_test = df['Survived']
    yield { "ACCURACY": model.score(X_test, y_test)}

# modelop.score
def predict(X):
    df = pd.DataFrame(X, index=[0])
    y_pred = model.predict(df)
    for p in y_pred:
        yield p



if __name__ == "__main__":
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    pred_df = pd.read_csv('predict.csv')

    train(train_df)
    begin()

    X = [[519,2,"Bob","male",36.0,1,0,226875,26.0,'C26',"S"]]
    print(predict(X))

    for m in metrics(test_df):
        print(m)
