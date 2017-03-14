import pandas as pd
from sklearn.model_selection import train_test_split

import house_prices.feature_eng as fe
from utils.plotter import plot_predict_accuracity
from utils.predict import select_model, predict_model
from utils.submit import submit

f = "C:\\data\\projects\\python\\data\\house_price\\"
df_test = pd.read_csv(f + "test.csv")
df_train = pd.read_csv(f + "train.csv")

print("Total Rows train={0}".format(len(df_train.columns)))
print("Total Rows test={0}".format(len(df_test.columns)))
print("Train data len={0}".format(len(df_train)))

x_train, x_test, y_train, y_test = train_test_split(df_train.drop("SalePrice", axis=1),
                                                    df_train["SalePrice"],
                                                    test_size=0.33, random_state=42)

crl = x_train
crl["SalePrice"] = y_train
corr_values = fe.numeric_correlation(crl, "SalePrice")

x_train = fe.eng(x_train, corr_values)
x_test = fe.eng(x_test, corr_values)

res = select_model(x_train, y_train)
plot_predict_accuracity(res["predict_train"], y_train, "Train")

predict_model(res["clf"], x_test, y_test)
plot_predict_accuracity(res["clf"].predict(x_test), y_test, "Test")

x_super_test = fe.eng(df_test, corr_values)
super_result = res["clf"].predict(x_super_test)

submit(df_test["Id"], super_result, f, "Id", "SalePrice")

# See other data columns (Maby not numeric)
# Numeric data with low correlation
# Remove high correlated data
# two level predictions
# Maybe prediction with several models (maybe not)
# try to fit other parameters of model
# grid cv=5
