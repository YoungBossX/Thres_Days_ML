from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def nb_news():
    """
    用朴素贝叶斯算法对新闻进行分来
    :return: None
    """
    # 获取数据
    news = fetch_20newsgroups(data_home="E:\Pycharm\Projects\Thres_Days\Day_2",subset='all')
    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)
    # 特征工程：文本抽取-tfidf
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 朴素贝叶斯算法预估器流程
    estimator = MultinomialNB()
    # 模型评估
    estimator.fit(x_train, y_train)
    # Predict the labels of the test set
    y_predict = estimator.predict(x_test)
    print("Predicted labels:\n", y_predict)
    print("直接比对真实值和预测值\n", y_test == y_predict)
    # Calculate the accuracy of the model
    score = estimator.score(x_test, y_test)
    print("Accuracy:\n", score)
    
    return None

if __name__ == "__main__":
    # nb_news()