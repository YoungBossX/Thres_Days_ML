from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error
import joblib

def linear_1():
    '''
    正规方程优化方法
    :rerurn: None
    '''
    # 获取数据
    california_housing = fetch_california_housing()
    print("特征数量：\n", california_housing.data.shape)
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(california_housing.data, california_housing.target, random_state=22)
    # 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 预估器
    estimator = LinearRegression()  
    estimator.fit(x_train, y_train)
    # 得出模型
    print('正规方程-权重系数为：\n', estimator.coef_)
    print('正规方程-偏置为：\n', estimator.intercept_)
    # 模型评估
    y_predict = estimator.predict(x_test)
    print('正规方程-预测值为：\n', y_predict)
    error = mean_squared_error(y_test, y_predict)
    print('正规方程-均方误差为：\n', error)   
    
    return None

def linear_2():
    '''
    梯度下降优化方法
    :rerurn: None
    '''
    # 获取数据
    california_housing = fetch_california_housing() 
    print("特征数量：\n", california_housing.data.shape)
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(california_housing.data, california_housing.target, random_state=22)
    # 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 预估器
    estimator = SGDRegressor(max_iter=10000, eta0=0.001,random_state=22)
    estimator.fit(x_train, y_train)
    # 得出模型
    print('梯度下降-权重系数为：\n', estimator.coef_)
    print('梯度下降-偏置为：\n', estimator.intercept_)
    # 模型评估
    y_predict = estimator.predict(x_test)
    print('梯度下降-预测值为：\n', y_predict)
    error = mean_squared_error(y_test, y_predict)
    print('梯度下降-均方误差为：\n', error)  
    
    return None

def linear_3():
    '''
    加上岭回归再做优化
    :rerurn: None
    '''
    # 获取数据
    california_housing = fetch_california_housing() 
    print("特征数量：\n", california_housing.data.shape)
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(california_housing.data, california_housing.target, random_state=22)
    # 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 预估器
    estimator = Ridge(alpha=0.5, max_iter=10000)
    estimator.fit(x_train, y_train)
    # 保存模型
    joblib.dump(estimator, 'ridge_model.pkl')
    
    # 得出模型
    print('岭回归-权重系数为：\n', estimator.coef_)
    print('岭回归-偏置为：\n', estimator.intercept_)
    # 模型评估
    y_predict = estimator.predict(x_test)
    print('岭回归-预测值为：\n', y_predict)
    error = mean_squared_error(y_test, y_predict)
    print('岭回归-均方误差为：\n', error)  
    
    return None

def load_model():
    # 获取数据
    california_housing = fetch_california_housing() 
    print("特征数量：\n", california_housing.data.shape)
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(california_housing.data, california_housing.target, random_state=22)
    # 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 加载预估器模型
    estimator = joblib.load('ridge_model.pkl')
    # 得出模型
    print('岭回归-权重系数为：\n', estimator.coef_)
    print('岭回归-偏置为：\n', estimator.intercept_)
    # 模型评估
    y_predict = estimator.predict(x_test)
    print('岭回归-预测值为：\n', y_predict)
    error = mean_squared_error(y_test, y_predict)
    print('岭回归-均方误差为：\n', error)  

if __name__ == '__main__':
    # linear_1()
    # linear_2()
    # linear_3()
    # load_model()
    