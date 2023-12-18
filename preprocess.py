import os
import csv

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

def write_csv(data):
    keys = data[0].keys()
    with open('dataset.csv', 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)

def main():
    all_data = []
    for category, label in categories.items():
        all_data.extend(process_category(category, label))

    write_csv(all_data)

if __name__ == "__main__":
    main()
