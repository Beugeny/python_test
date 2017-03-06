import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures

import utils.plotter

f = "C:\\data\\projects\\machineLearning\\kaggle\\competitions\\titanic\\"
train = pd.read_csv(f + "train.csv")
ptrain = pd.DataFrame()

ptrain['Sex'] = train["Sex"].map({"male": 1, "female": 0}).astype(int)
ptrain['Family'] = train['Parch'] + train['SibSp']

ageTrainData = train[["Pclass", "SibSp", "Parch", "Fare", "Age","Survived"]]
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

utils.plotter.plot_predict_accuracity(clf.predict(ageX.values), ageY.values)
