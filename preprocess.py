import os
import csv
from sklearn.model_selection import train_test_split

# Define the categories and their labels
categories = {
    "damaged_infrastructure": 0,
    "damaged_nature": 1,
    "fires": 2,
    "flood": 3,
    "human_damage": 4,
    "non_damage": 5
}

def read_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def process_category(category, label):
    image_dir = f"./multimodal/{category}/images"
    text_dir = f"./multimodal/{category}/text"
    data = []

    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            base_filename = filename[:-4]
            text_path = os.path.join(text_dir, base_filename + ".txt")
            image_path = os.path.join(image_dir, filename)

            if os.path.exists(text_path):
                text = read_text(text_path)
                data.append({
                    "text": text,
                    "textlocation": text_path,
                    "imagelocation": image_path,
                    "Label": label,
                    "Filename": base_filename,
                    "damagekind": category,
                    # Assuming NLPscores, CVscores, IRscores, WAscores, RFscores are calculated elsewhere
                    "NLPscores": "",
                    "CVscores": "",
                    "IRscores": "",
                    "WAscores": "",
                    "RFscores": ""
                })
    return data



def main():
    all_data = []
    for category, label in categories.items():
        all_data.extend(process_category(category, label))

    # 将数据切分为训练集和测试集
    train_data, test_data = train_test_split(all_data, test_size=0.1, random_state=42)

    # 分别写入训练集和测试集数据到CSV文件
    write_csv(train_data, './TrainTest/train_output.csv')
    write_csv(test_data, './TrainTest/test_output.csv')

def write_csv(data, filename):
    keys = data[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)

if __name__ == "__main__":
    main()
