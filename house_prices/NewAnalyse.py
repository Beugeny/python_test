import pandas as pd

f = "C:\\data\\projects\\python\\data\\house_price\\"
df_test = pd.read_csv(f + "test.csv")
df_train = pd.read_csv(f + "train.csv")


# TODO
# Анализ данных
# 1. Нужно происследовать каждый признак и попытатьс ялогически понять как его можно преобразовать или заполнить nan
# 1. P=Получить все не категориальные признаки
# 2. P+=Получить и преобразовать все категориальные признаки pd.get_dummies()
# 2. P=удалить выбросы P np.percentile
# 2. K=Найти корреляции P по отношению к целевому
# 3. K2=Найти корреляции P отношению друг к другу
# 4. NAN=Отсортированый по убыванию список столбцов P, с максимальным содержанием пустых данных
# 5. RESP=Удалить лишние столбцы из P по критерию: низкая K, высокая K2, много NAN
# 6. Заполнить пустые данные в RESP. Handling Missing Data Kernel
# 7. Исследовать данные на Normality. ТОесть соответстиве нормальному закону. (Не до конца понимаю зачем)
# 7. Исследовать данные на Homoscedasticity. Тоесть одинаковость разнообразия (Не до конца понимаю зачем)
# 7. Исследовать данные на Linearity Что взаимодейстиве линейное(Не до конца понимаю зачем)

# Построение модели (Лучше это сделать некоторым независимым алгоритмом)
# 1. Стандартизация признаков
# 1. Разделение на 3 выборки test train cv  используя f_fold
# 1.1 для каждого потенциального алгоритма выполняем grid_search. (Используем класс оболочка для классификаторов)
# 1.2 K[i]=находим наилучший алгоритм по оценке на cv выборке
# 2. Из K Получаем оценку алгоритма на cv выборке - среднее значение и диспресия


# Реализация двух уровневой модели - когда на первом уровне мы реализуем предсказание
# На нескольких алгоритмах сразу, а на втором на основе предсказаний предсказываем результат

# Конец
# 1. Предсказываем на test - находим результат на них. Оценка по идее должна совпадать с оценкой на Kaggle
# 2  Предсказываем на релизных данных - submit
