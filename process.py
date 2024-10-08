import torch
import json

def process_data(data_file):
    # 实现数据处理逻辑，例如分词、标签编码等

    with open(data_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Initialize a dictionary to hold unique labels and their counts
    label_ids = {}
    label_ids['0'] = 0
    # Iterate over each entry in the JSON data
    i = 1
    for entry in data:
        if "annotations" in entry:
            for annotation in entry["annotations"]:
                label = annotation["label"]
                if label not in label_ids:
                    label_ids[label] = i
                    i += 1
                  # Count occurrences of each label

    # Create label.txt and write the labels and their counts
    with open("Data/label.txt", "w", encoding="utf-8") as f:
        for label, count in label_ids.items():
            f.write(f"{label}:{count}\n")  # Write label and count

    print("Labels and counts extracted and saved to label.txt.")

    with open(data_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Load the label to id mapping from label.txt
    label_to_id = {}
    with open("D:\\Deep_learn_NER\\Data\\label.txt", "r", encoding="utf-8") as f:
        for line in f:
            label, label_id = line.strip().split(':')
            label_to_id[label] = int(label_id)

    # Prepare the list for the new format
    output_list = []
    # Process each entry in the JSON data
    for entry in data:
        if "text" in entry and "annotations" in entry:
            text = entry["text"]
            text = list(text)
            label_array = ['0'] * len(text)  # Initialize all positions to 0

            # Populate the label array based on annotations
            for annotation in entry["annotations"]:
                label = annotation["label"]
                start_offset = annotation["start_offset"]
                end_offset = annotation["end_offset"]

                    # Set the corresponding position in the label array
                for i in range(start_offset, end_offset):
                    label_array[i] = label  # Set to label_id at the correct position

            # Create a dictionary for this entry
            entry_dict = {
                "text": text,
                "label": label_array
            }
            for i in range(0, len(text), 510):
                tmp_text = text[i:i+510]
                tmp_label = label_array[i:i+510]
                entry_dict = {
                    "text": tmp_text,
                    "label": tmp_label
                }
                output_list.append(entry_dict)

    # Save the output list to a new JSON file
    with open("Data/output_list.json", "w", encoding="utf-8") as f:
        json.dump(output_list, f, ensure_ascii=False, indent=4)

    print("Data transformed and saved to output_list.json.")

if __name__ == '__main__':

    data_file = "D:\\Deep_learn_NER\\Data\\medical_ner_entities.json"
    process_data(data_file)
