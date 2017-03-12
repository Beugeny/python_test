import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils.featEng import isExist

f = "D:\\resources\\machineLearning\\python_test\\data\\"
allData = pd.read_csv(f + "air_crashes.csv")
X = pd.DataFrame()

print(allData.columns.values)
print(allData["Time"].head())

# Parse date
f = "%m/%d/%Y"
dates = [datetime.strptime(x, f) for x in allData["Date"]]
X["Y"] = pd.Series([x.year for x in dates])
X["M"] = pd.Series([x.month for x in dates])
X["D"] = pd.Series([x.day for x in dates])

# parse has time
X["HAS_TIME"] = pd.Series([isExist(x) for x in allData["Time"]])


# parse location
#
# def getLocation(x, index):
#     if isExist(x):
#         tmp = x.split(",")
#         if len(tmp) > index:
#             return tmp[index]
#         else:
#             return None
#     else:
#         return None
#
#
# LOCS = pd.DataFrame()
# LOCS["L"] = pd.Series([getLocation(x, 0) for x in allData["Location"]])
# LOCS["L"].append(pd.Series([getLocation(x, 1) for x in allData["Location"]]))
# LOCS["L"].append(pd.Series([getLocation(x, 2) for x in allData["Location"]]))
# LOCS["L"].append(pd.Series([getLocation(x, 3) for x in allData["Location"]]))
# unicLocations=pd.Series(LOCS["L"].unique())
# print("UNic count:={0}".format(len(unicLocations)))
# print(unicLocations.sort_values())
# #
# counts = []
# for x in X["Y"].unique():
#     counts.append(len(X.loc[X["Y"] == x].values))
# plt.plot(counts)
# plt.show()
# print(len(counts), np.mean(counts))



print(X.head())
