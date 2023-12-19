from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score,confusion_matrix
import pandas as pd
file_path = './TrainTest/test_dataset.csv'

# 读取CSV文件
test_df = pd.read_csv(file_path)

metrics_dict = {}
model_columns = ['NLP', 'CV', 'LinearRegression', 'RandomForest', 'LogisticRegression', 'SVC', 'WeightedAverage']

# 计算各项指标
for model in model_columns:
    metrics_dict[model] = {
        'Accuracy': accuracy_score(test_df['Label'], test_df[model]),
        'Precision': precision_score(test_df['Label'], test_df[model], average='weighted'),
        'Recall': recall_score(test_df['Label'], test_df[model], average='macro'),
        'F1 Score': f1_score(test_df['Label'], test_df[model], average='weighted')
            }

# 将字典转换为数据框架
metrics_df = pd.DataFrame(metrics_dict).transpose()

# 显示数据框架
print(metrics_df)
