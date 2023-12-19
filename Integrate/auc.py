import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# 设置数据集的文件路径
file_path = './TrainTest/test_dataset.csv'

# 读取CSV文件
test_df = pd.read_csv(file_path)

# 假设 'Label' 列是真实标签，并且所有的模型预测列都已经是二进制的类别预测
# 计算每个模型的F1分数
model_columns = ['NLP', 'CV', 'LinearRegression', 'RandomForest', 'LogisticRegression', 'SVC', 'WeightedAverage']
f1_scores = {}

for model in model_columns:
    f1_scores[model] = f1_score(test_df['Label'], test_df[model], average='micro')  # 使用'micro'来计算总体的F1分数

# 将F1分数转换为数据框架，以便于绘图
f1_scores_df = pd.DataFrame(list(f1_scores.items()), columns=['Model', 'F1 Score'])

# 绘制F1分数图表
plt.figure(figsize=(10, 6))
barplot = sns.barplot(x='Model', y='F1 Score', data=f1_scores_df)
plt.title('F1 Scores for Different Models')
plt.xticks(rotation=45)
plt.ylabel('F1 Score')
plt.xlabel('Model')

# 在条形上添加分数标签
for p in barplot.patches:
    barplot.annotate(format(p.get_height(), '.1%'),  # 转换为百分比格式
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha = 'center', va = 'center',
                     xytext = (0, 9),
                     textcoords = 'offset points')

plt.tight_layout()  # 确保标签不会重叠
plt.show()

# 输出计算的F1分数，以便检查
f1_scores_df
