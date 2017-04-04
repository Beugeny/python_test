import pandas as pd

from utils.plotter import corr_zoom


def numeric_correlation(df, targetName):
    df = pd.DataFrame(df)
    num_columns = df[df.columns[df.dtypes != object]]

    cor = num_columns.corr()
    cor = abs(cor)
    cor2 = cor.apply(lambda s: s.apply(lambda x: (x if x > 0.5 else 0)))
    # print(cor2)
    # print(cor[targetName].sort_values(ascending=False))
    corr_zoom(cor, df, "SalePrice", 12)
    return cor[targetName].sort_values(ascending=False)


def eng(df, corr_values):
    # print("Feature engineering")
    # Этот метод не должен удалять ни один столбец
    df = pd.DataFrame(df)
    res = pd.DataFrame()

    # Удаляем лишние и высоко коррелированные столбцы( дубликаты данных), а
    # также столбцы где >15% данных отсутствует
    filter_set = set(
        ["SalePrice", "Id", "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "LotFrontage", "GarageYrBlt",
         'TotRmsAbvGrd'])


    # TODO Нормализация данных (изучить мат часть),
    # TODO Homoscedasticity, Linearity,Absence of correlated errors  Ooooooo!!!!!!!!
    # TODO хорошенько провести удаление коррелированных столбцов(автоматическое) и столбцов с пустыми переменными
    # TODO dummy variables WTF?

    for column in corr_values.index.values:
        if column not in filter_set and corr_values[column] > 0.1:
            # print(column, corr_values[column])
            if len(df[column]) != len(df[column].dropna()):
                # print('"{0}" column contain {1} nan values'.format(column, len(df[column]) - len(df[column].dropna())))
                df[column] = df[column].fillna(df[column].mean())  # TODO smart fill missing item
            res[column] = df[column]

    # poly = PolynomialFeatures(3, include_bias=False)
    # res = poly.fit_transform(res)
    # res = poly.fit_transform(res)

    # scaler = preprocessing.StandardScaler().fit(res)
    # res = scaler.transform(res)

    return res
