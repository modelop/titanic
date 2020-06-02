from sklearn.model_selection import train_test_split
import pandas as pd


df = pd.read_csv('titanic.csv')


train, test = train_test_split(df, test_size=0.2)


train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)