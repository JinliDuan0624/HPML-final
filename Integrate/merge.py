import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# 设置文件路径
train_file_path = './TrainTest/train_dataset.csv'
test_file_path = './TrainTest/test_dataset.csv'

# 加载数据集
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

def str_to_list(str_list):
    return ast.literal_eval(str_list)
train_df['NLPscores'].fillna('[0,0,0,0,0,0]', inplace=True)
train_df['CVscores'].fillna('[0,0,0,0,0,0]', inplace=True)
test_df['NLPscores'].fillna('[0,0,0,0,0,0]', inplace=True)
test_df['CVscores'].fillna('[0,0,0,0,0,0]', inplace=True)
# 转换 NLPscores 和 CVscores 列
# 转换 NLPscores 和 CVscores 列为实际的数值列表
train_df['NLPscores'] = train_df['NLPscores'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
train_df['CVscores'] = train_df['CVscores'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# 确保 X_train 的长度与 y_train 的长度相同
X_train = [nlp + cv for nlp, cv in zip(train_df['NLPscores'], train_df['CVscores'])]
y_train = train_df['Label'].values

# 确认 X_train 和 y_train 的长度相等
assert len(X_train) == len(y_train), "The lengths of X_train and y_train must be equal."


# 对测试集进行同样的转换
test_df['NLPscores'] = test_df['NLPscores'].apply(str_to_list)
test_df['CVscores'] = test_df['CVscores'].apply(str_to_list)
X_test = [nlp + cv for nlp, cv in zip(test_df['NLPscores'], test_df['CVscores'])]
y_test = test_df['Label'].values

# 定义和训练模型
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'SVC': SVC(probability=True)
}

# 训练模型并计算准确率
for model_name, model in models.items():
    # 训练模型
    model.fit(X_train, y_train)

    # 如果是线性回归模型，输出需要四舍五入到最近的整数
    if model_name == 'LinearRegression':
        y_pred = np.rint(model.predict(X_test)).astype(int)
    else:
        y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)

    # 打印准确率
    print(f'{model_name} Accuracy: {accuracy}')

    # 将预测结果保存到测试数据集
    test_df[model_name] = y_pred

# 计算加权平均分数
nlp_weight = 0.88  # NLP模型的权重
cv_weight = 0.80  # CV模型的权重
# 定义一个新的函数来处理可能的数据问题
def calculate_weighted_average(nlp_scores, cv_scores, nlp_weight, cv_weight):
    weighted_scores = [nlp_weight * n + cv_weight * c for n, c in zip(nlp_scores, cv_scores)]
    return weighted_scores.index(max(weighted_scores))  # 返回最大值的索引
test_df['WeightedAverage'] = test_df.apply(
    lambda row: calculate_weighted_average(row['NLPscores'], row['CVscores'], nlp_weight, cv_weight), axis=1)
def calculate_max_index(scores):
    return scores.index(max(scores))

# 应用 calculate_max_index 函数到每一行
test_df['NLP'] = test_df['NLPscores'].apply(calculate_max_index)
test_df['CV'] = test_df['CVscores'].apply(calculate_max_index)
# 计算加权平均的准确率
weighted_average_accuracy = accuracy_score(test_df['Label'], test_df['WeightedAverage'])
print(f'Weighted Average Accuracy: {weighted_average_accuracy}')

# 保存测试数据集
test_df.to_csv(test_file_path, index=False)
