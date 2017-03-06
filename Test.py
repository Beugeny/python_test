import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures



A=[1,2,3,4,5,6]
B=[10,20,30,40,50,60]


df = pd.DataFrame([A,B])
df=df.T
df.columns=['A','B']
print(df)

poly = PolynomialFeatures(2)
C=poly.fit_transform(df)

np.set_printoptions(precision=3,suppress=True)


print(C)
print(df)

