# titanic


1. on-board model to MOC
2. map functions.  init->begin, score->predict, metrics->metrics, train->train

use test.csv for a metrics job
use predeict.csv for a scoring job


to test inference using the rest api:
```
import requests

inference = {'PassengerId': 519, 'Pclass': 2, 'Name': 'Bob', 'Sex': 'male', 'Age': 36, 'SibSp': 0,
             'Parch':1,'Ticket': 226875, 'Fare': 26.0,'Cabin': 'C26', 'Embarked': 'S'}   

response = requests.post('{gateway-url}/{engine-name}/api/roundtrip/0/1', json=inference)

print("Status code: ", response.status_code)
print("Printing Entire Post Request")
print(response.json())
```

