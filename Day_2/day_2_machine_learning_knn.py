from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def knn_iris():
    """
    K-Nearest Neighbors (KNN) algorithm on the Iris dataset.
    :return:
    """
    # Load the Iris dataset
    iris = load_iris()
    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22) # random_state能影响准确率
    # Standardize the features
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # Create a KNN classifier
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)
    # Predict the labels of the test set
    y_predict = estimator.predict(x_test)
    print("Predicted labels:\n", y_predict)
    print("直接比对真实值和预测值\n", y_test == y_predict)
    # Calculate the accuracy of the model
    score = estimator.score(x_test, y_test)
    print("Accuracy:\n", score)
    
    return None

def knn_iris_gscv():
    """
    K-Nearest Neighbors (KNN) algorithm on the Iris dataset with GridSearchCV.
    :return:
    """
    # Load the Iris dataset
    iris = load_iris()
    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22) # random_state能影响准确率
    # Standardize the features
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # Create a KNN classifier
    estimator = KNeighborsClassifier()
    # 加入网格搜索
    # Define the parameter grid for GridSearchCV
    param_grid = {
        "n_neighbors": [1, 3, 5, 7, 9, 11],
    }
    # Create a GridSearchCV object
    estimator = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=10)
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
    
    return None

if __name__ == "__main__":
    # knn_iris()
    # knn_iris_gscv()