from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split

def decision_iris():
    """
    用决策树对鸢尾花进行分类
    :return: None
    """
    # 获取数据集
    iris = load_iris()
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)
    # 特征工程（选做）

    # 决策树预估器
    estimator = DecisionTreeClassifier(criterion='entropy')
    estimator.fit(x_train, y_train)
    # 模型评估
    y_predict = estimator.predict(x_test)
    print("Predicted labels:\n", y_predict)
    print("直接比对真实值和预测值\n", y_test == y_predict)
    score = estimator.score(x_test, y_test)
    print("Accuracy:\n", score)
    # 可视化决策树
    export_graphviz(estimator, out_file='iris_tree.dot', feature_names=iris.feature_names)

    return None

if __name__ == '__main__':
    # decision_iris()