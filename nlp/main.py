import sys
import pandas as pd
from data_preparation import load_and_preprocess_data
from train_and_evaluate import train_and_evaluate
from predict import load_model, predict
from sklearn.metrics import accuracy_score


def batch_predict(model, tokenizer, descriptions):
    return [predict(desc, model, tokenizer) for desc in descriptions]
def evaluate(model, tokenizer, descriptions, true_labels):
    predictions = batch_predict(model, tokenizer, descriptions)
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy


def main():
    currentdataset='train'
    dataset_path = "./dataset/TrainTest/"+currentdataset+"_dataset.csv"
    model_path = "trained_model"
    num_labels = 6  # 根据你的任务调整这个值

    # 检查命令行参数来确定执行哪个部分
    if len(sys.argv) < 2:
        print("Usage: python main.py [prepare/train/predict]")
        sys.exit(1)

    if sys.argv[1] == "prepare":
        print("Preparing data...")
        load_and_preprocess_data(dataset_path)

    elif sys.argv[1] == "train":
        print("Training model...")
        train_and_evaluate(dataset_path, model_path, num_labels)

    elif sys.argv[1] == "predict":
        # 现在不需要额外的命令行参数了
        print("Predicting...")
        model, tokenizer = load_model(model_path)

        # 读取 CSV 文件
        test_dataset = pd.read_csv(dataset_path)
        descriptions = test_dataset['text'].tolist()
        true_labels = test_dataset['Label'].tolist()  # 确保CSV文件有一个名为'label'的列，包含真实标签

        # 计算准确率
        accuracy = evaluate(model, tokenizer, descriptions, true_labels)
        print(f"Test Accuracy: {accuracy}")

        # 进行批量预测
        predictions = batch_predict(model, tokenizer, descriptions)

        # 将预测结果添加到 DataFrame
        test_dataset['NLPscores'] = predictions

        # 保存更新后的 DataFrame 到新的 CSV 文件
        test_dataset.to_csv('predictions.csv', index=False)
        print("Predictions saved to predictions.csv")

    else:
        print("Invalid argument. Use 'prepare', 'train' or 'predict'.")

if __name__ == "__main__":
    main()
