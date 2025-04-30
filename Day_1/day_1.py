import jieba
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA  
import pandas as pd
from scipy.stats import pearsonr

#************************************************************特征抽取************************************************************
def dataset_demo():
    """
    sklearn.datasets数据集使用
    :return:
    """
    iris = load_iris()
    print("鸢尾花数据集：\n", iris)
    print("查看数据集描述：\n", iris.DESCR)
    print("查看特征值得名字：\n", iris.feature_names)
    print("查看特征值：\n", iris.data, "\n", iris.data.shape)
    return None
    
def dataset_split():
    """
    sklearn.datasets数据集划分
    :return:
    """
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    print("训练集特征值：\n", X_train, "\n", X_train.shape)
    print("测试集特征值：\n", X_test, "\n", X_test.shape)
    print("训练集标签值：\n", y_train, "\n", y_train.shape)
    print("测试集标签值：\n", y_test, "\n", y_test.shape)
    return None
    
def dict_demo():
    """
    字典特征抽取
    :return:
    """
    data = [{'city': '北京', 'temperature': 20}, {'city': '上海', 'temperature': 25}, {'city': '杭州', 'temperature': 30}]
    transfer = DictVectorizer(sparse=False, )
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    print("特征值名字：\n", transfer.get_feature_names_out())
    
    return None

def text_demo():
    """
    文本特征抽取
    :return:
    """
    data = ["life is short ,i like like python", "life is too long , i dislike python"]
    transfer = CountVectorizer(stop_words=["is", "too"])
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray())
    print("特征值名字：\n", transfer.get_feature_names_out())
    
    return None

def text_demo_zh():
    """
    中文文本特征抽取：把短语当特征
    :return:
    """
    data = ["我 在 杭州电子科技大学", "现在 是 研一"]
    transfer = CountVectorizer()
    data_new = transfer.fit_transform(data).toarray()
    print("data_new:\n", data_new)
    print("特征值名字：\n", transfer.get_feature_names_out())
    
    return None

def cut_words(sentence):
    """
    中文分词函数
    :param sentence: 需要分词的句子
    :return: 分词后的结果
    """
    return " ".join(list(jieba.cut(sentence)))

def text_demo_zh_auto():
    """
    中文文本特征抽取，自动分词，可用找合适的分词工具进行自动分词
    :return:
    """
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    daea_new = []
    for sent in data:
        daea_new.append(cut_words(sent))
        
    transfer = CountVectorizer(stop_words=["一种", "所以"])
    
    data_final = transfer.fit_transform(daea_new)
    print("data_new:\n", data_final.toarray())
    print("特征值名字：\n", transfer.get_feature_names_out())
    
    return None

def tfidf_demo():
    """
    tf-idf特征抽取
    :return:
    """
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    daea_new = []
    for sent in data:
        daea_new.append(cut_words(sent))
    
    transfer = TfidfVectorizer(stop_words=["一种", "所以"])
    
    data_final = transfer.fit_transform(daea_new)
    print("data_new:\n", data_final.toarray())
    print("特征值名字：\n", transfer.get_feature_names_out())
    
    return None


#************************************************************特征预处理************************************************************
def minmax_demo():
    """
    特征预处理：归一化
    鲁棒性很差，最大值、最小值容易受异常值影响，适用于小数据，做无量纲化
    :return:
    """
    data = pd.read_csv("Thres_Days\Day_1\dating.txt")
    data = data.iloc[:, :3]
    print("data:\n", data)
    
    transfer = MinMaxScaler(feature_range=(0, 1))
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    
    return None

def standard_demo():
    """
    特征预处理：标准化
    鲁棒性较好，适用于大数据
    :return:
    """
    data = pd.read_csv("Thres_Days\Day_1\dating.txt")
    data = data.iloc[:, :3]
    print("data:\n", data)
    
    transfer = StandardScaler()
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    
    return None

#************************************************************特征降维************************************************************
def variance_demo():
    """
    特征降维：方差选择法
    :return:
    """
    data = pd.read_csv(r"Thres_Days\Day_1\factor_returns.csv")
    data = data.iloc[:, 1:-2]
    print("data:\n", data)
    
    transfer = VarianceThreshold(threshold=10)
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new, "\n", data_new.shape)
    
    # 计算某两个变量之间的相关系数
    # pearsonr_value = pearsonr(data["pe_ratio"], data["pb_ratio"])
    # print("pe_ratio和pb_ratio之间的相关系数：\n", pearsonr_value[0])
    
    return None

def pearsonr_demo():
    """
    皮尔逊相关系数
    :return:
    """
    data = pd.read_csv(r"Thres_Days\Day_1\factor_returns.csv")
    print("data:\n", data)
    
    factor = ['pe_ratio', 'pb_ratio', 'market_cap', 'return_on_asset_net_profit', 'du_return_on_equity', 'ev', 'earnings_per_share', 'revenue', 'total_expense']
    for i in range(len(factor)):
        for j in range(i + 1, len(factor)):
            pearsonr_value = pearsonr(data[factor[i]], data[factor[j]])
            print(f"{factor[i]}和{factor[j]}之间的相关系数：\n", pearsonr_value[0])
    
    return None

def pca_demo():
    """
    主成分分析法
    :return:
    """
    data = [[2, 8, 4, 5],
            [6, 3, 0, 8],
            [5, 4, 9, 1]] # 4个特征变为2个特征
    print("data:\n", data)
    
    transfer = PCA(n_components=2) # 0.95表示保留95%的信息
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    
    return None


#************************************************************案例分析************************************************************



if __name__ == "__main__":
    # dataset_demo()
    # dataset_split()
    # dict_demo()
    # text_demo()
    # text_demo_zh()
    # text_demo_zh_auto()
    # tfidf_demo()
    # minmax_demo()
    # standard_demo()
    # variance_demo()
    # pearsonr_demo()
    # pca_demo()
    
