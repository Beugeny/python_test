import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import utils.plotter

f = "C:\\data\\projects\\machineLearning\\kaggle\\competitions\\titanic\\"
train = pd.read_csv(f + "train.csv")
pData = pd.DataFrame()

print(train.head(3))

pData["Pclass"] = train["Pclass"]
pData["Fare"] = train["Fare"]
pData['Sex'] = train["Sex"].map({"male": 1, "female": 0}).astype(int)
pData['Family'] = train['Parch'] + train['SibSp']
pData["Survived"] = train["Survived"]

ageTrainData = train[["Pclass", "SibSp", "Parch", "Fare", "Age", "Survived"]]
nan_age = np.isnan(ageTrainData["Age"])
ageTrainData = ageTrainData.dropna()

ageX = ageTrainData.drop("Age", axis=1)
ageY = ageTrainData["Age"]

# poly = PolynomialFeatures(2,include_bias=False)
# A=poly.fit_transform(ageX)
# ageX=pd.DataFrame(A)


clf = DecisionTreeRegressor()
clf.fit(ageX.values, ageY.values)
print(clf.score(ageX.values, ageY.values))

exp = ageY.values
predicted = clf.predict(ageX.values)
# utils.plotter.plot_predict_accuracity(clf.predict(ageX.values), ageY.values)

res = clf.predict(ageX.values)
pData["Age"] = train["Age"]

oldValues = pData["Age"].dropna().values

# X1 = ptrain.dropna()["Fare"].values
# Y1 = ptrain.dropna()["Age"].values;

i = 0
ageValues = pData["Age"].values
for index, val in enumerate(ageValues):
    if np.isnan(val):
        ageValues[index] = res[i]
        i += 1
pd.DataFrame.replace(pData["Age"], list(ageValues))

# utils.plotter.show_two_hist(ptrain["Age"].dropna().values, oldValues)
print(pData.head(15))
print("Has nan values={0}".format(len(pData) > len(pData.dropna())))



# CLASSIFICATION

X_train, X_test, y_train, y_test = train_test_split(pData.drop("Survived", axis=1), pData["Survived"], test_size=0.33,
                                                    random_state=42)

# Чтото полиномиальные фичи не на что не влияют(((
# poly = PolynomialFeatures(2,include_bias=False)
# A=poly.fit_transform(X_train)
# X_train=pd.DataFrame(A)
#
# poly = PolynomialFeatures(2,include_bias=False)
# A=poly.fit_transform(X_test)
# X_test=pd.DataFrame(A)


clf = DecisionTreeClassifier(random_state=False)
clf.fit(X_train, y_train)

train_result = clf.predict(X_train)
target_names = ['Survived', 'Not survived']

print("Train result")
print(classification_report(y_train, train_result, target_names=target_names))
print(accuracy_score(y_train, train_result))

print("Test result")
test_result = clf.predict(X_test)
print(classification_report(y_test, test_result, target_names=target_names))
print(accuracy_score(y_test, test_result))
