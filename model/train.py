import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import joblib

from data.species import species

# load dataset
data = pd.read_csv("../data/iris.csv")
# print(data.head())
# print(data['species'].value_counts())

# map species to numerical values
data["species"] = data["species"].map(species)

# print(data['species'].value_counts())

# define features and labels (X and y)
X = data.drop("species", axis=1)
y = data["species"]

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# initialize and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)
# print(y_pred)

# model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

joblib.dump(model, filename="../model/model.pkl")
