import pandas as pd
from sklearn import preprocessing


def numeric_correlation(df, targetName):
    df = pd.DataFrame(df)
    num_columns = df[df.columns[df.dtypes != object]]

    cor = num_columns.corr()
    cor = abs(cor)
    print(cor[targetName].sort_values(ascending=False))
    return cor[targetName].sort_values(ascending=False)


def eng(df, corr_values):
    print("Feature engineering")
    # Этот метод не должен удалять ни один столбец
    df = pd.DataFrame(df)
    res = pd.DataFrame()

    for column in corr_values.index.values:
        if column != "SalePrice" and column != "Id" and corr_values[column] > 0.1:
            print(column, corr_values[column])
            if len(df[column]) != len(df[column].dropna()):
                print('"{0}" column contain {1} nan values'.format(column, len(df[column]) - len(df[column].dropna())))
                df[column] = df[column].fillna(0)  # TODO fill with fit value
            res[column] = df[column]


    # poly = PolynomialFeatures(3, include_bias=False)
    # res = poly.fit_transform(res)
    # res = poly.fit_transform(res)

    # scaler = preprocessing.StandardScaler().fit(res)
    # res = scaler.transform(res)

    return res
