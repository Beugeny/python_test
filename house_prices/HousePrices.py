import pandas as pd

import house_prices.feature_eng as fe
from utils.plotter import plot_predict_accuracity
from utils.predict import fit_predict

f = "D:\\resources\\machineLearning\\python_test\\data\\house_prices\\"
df_test = pd.read_csv(f + "test.csv")
df_train = pd.read_csv(f + "train.csv")

print(df_train.columns.values)
print(len(df_train))

f_data = fe.eng(df_train)

res = fit_predict(f_data.drop("SalePrice",axis=1), f_data["SalePrice"])

plot_predict_accuracity(res["predict_train"], res["y_train"], "Train")
plot_predict_accuracity(res["predict_test"], res["y_test"], "Test")
