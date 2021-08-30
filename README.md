# titanic

A Model to predict likelihood of Survival on board the ill-fated Titanic.

Model code contains scoring (prediction), metrics (accuracy), and training functions.

## Scoring Jobs

### Sample Inputs

Choose the following file for a sample scoring job:
 - `predict.csv`

### Schema Checking

Schema Checking is **unavailable** (no schemas exist for this model).

### Sample Output

The output of the scoring job when the input data is `predict.csv` is a JSONS file (one-line JSON records). Here are the first two output records:
```json
{"Ticket": 349248, "SibSp": 0, "Sex": "male", "Pclass": 3, "PassengerId": 871, "Parch": 0, "Name": "Balkic, Mr. Cerin", "Fare": 7.8958, "Embarked": "S", "Cabin": null, "Age": 26.0, "Prediction": 0}
{"Ticket": 113781, "SibSp": 1, "Sex": "female", "Pclass": 1, "PassengerId": 499, "Parch": 2, "Name": "Allison, Mrs. Hudson J C (Bessie Waldo Daniels)", "Fare": 151.55, "Embarked": "S", "Cabin": "C22 C26", "Age": 25.0, "Prediction": 1}
```

## Metrics Jobs

Model code includes a metrics function used to compute accuracy.

### Sample Inputs

Choose the following file for a sample metrics job:
 - `test.csv`


## Training Jobs

Model Code includes a training function used to train a model binary.

### Sample Inputs

Choose **one** of:
 - `train.csv`
 - `train.json`

### Output Files

In order to be able to download the pickle file that is written by the training function, add the following file as output to the training job:
 - `RFC_model.pkl`