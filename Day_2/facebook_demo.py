import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# print(f"当前工作目录: {os.getcwd()}")

# 读书数据
data = pd.read_csv("Day_2/FBlocation/train.csv")
print("data:\n", data.head())
print("data.shape:\n", data.shape)

# 缩小范围
data = data.query("x<2.5 & x>2.0 & y<2.5 & y>1.0")
print("data:\n", data.head())
print("data.shape:\n", data.shape)

# 处理时间特征
time_value = pd.to_datetime(data["time"], unit="s")
print("time_value:\n", time_value.head())

date = pd.DatetimeIndex(time_value)
data['day'] = date.day
data['weekday'] = date.weekday
data['hour'] = date.hour
print("date.year:\n", date.year)
print("date.month:\n", date.month)
print("date.day:\n", date.day)
print("date.weekday:\n", date.weekday)
print("date.hour:\n", date.hour)

# 过滤签到次数少的样本
place_count = data.groupby("place_id").count()["row_id"]    
print("place_count:\n", place_count)

place_count = place_count[place_count > 3]
print("place_count:\n", place_count.head())
 
data_final = data[data["place_id"].isin(place_count[place_count > 3].index.values)]
print("data_final:\n", data_final.head())

# 筛选特征值和目标值
x = data_final[["x", "y", "accuracy", "day",  "weekday", "hour"]]
y = data_final["place_id"]

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Standardize the features
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
# Create a KNN classifier
estimator = KNeighborsClassifier()
# Define the parameter grid for GridSearchCV
param_grid = {
    "n_neighbors": [3, 5, 7, 9],
}
# Create a GridSearchCV object
estimator = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=3)
# Fit the model to the training data
estimator.fit(x_train, y_train)
# Predict the labels of the test set
y_predict = estimator.predict(x_test)
print("Predicted labels:\n", y_predict)
print("直接比对真实值和预测值\n", y_test == y_predict)
# Calculate the accuracy of the model
score = estimator.score(x_test, y_test)
print("Accuracy:\n", score)
# Best parameters, score, estimator, and cv_results
print("Best parameters:\n", estimator.best_params_)
print("Best score:\n", estimator.best_score_)
print("Best estimator:\n", estimator.best_estimator_)
print("CV results:\n", estimator.cv_results_)