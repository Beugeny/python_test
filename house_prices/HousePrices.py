import pandas as pd

import house_prices.feature_eng as fe
from utils.plotter import plot_predict_accuracity
from utils.predict import regression_fit
from utils.submit import submit

f = "D:\\resources\\machineLearning\\python_test\\data\\house_price\\"
df_test = pd.read_csv(f + "test.csv")
df_train = pd.read_csv(f + "train.csv")

print("Total Rows={0}".format(df_train.columns.values))
print(len(df_train))

f_data = fe.eng(df_train)

res = regression_fit(f_data, df_train["SalePrice"])

plot_predict_accuracity(res["predict_train"], res["y_train"], "Train")
plot_predict_accuracity(res["predict_test"], res["y_test"], "Test")

f_test_data = fe.eng(df_test)
final_res = res["clf"].predict(f_test_data)

submit(df_test["Id"], final_res, f, "Id", "SalePrice")
