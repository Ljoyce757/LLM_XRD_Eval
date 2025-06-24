import csv
import json
# === To Import CSV Data into JSON ===
def read_json_file(file_path):
    """
    Opens a JSON file and reads its contents into a dictionary.
    Returns an empty dictionary if the file does not exist or is empty.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def read_csv_file(file_path):
    """
    Opens a CSV file and reads its contents into a list of dictionaries.
    Each dictionary represents a row with column headers as keys.
    """
    data = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

def csv_info_to_json(csv_file_path, json_file_path):
    csv_data = read_csv_file(csv_file_path)
    json_data = read_json_file(json_file_path)

    for row in csv_data: #iterate over each row in the CSV file
        run_name = row["Run Name in ALAB"].strip()
        if run_name not in json_data:
            json_data[run_name] = {}
        # Populate the JSON data with the relevant fields from the CSV
        json_data[run_name]["Synth_Conditions"] = {
            "Target": row["Target"],
            "Precursor 1": row["Precursor 1"],
            "Precursor 2": row["Precursor 2"],
            "Precursor 3": row["Precursor 3"],
            "Furnace": row["Furnace"],
            "Temperature (C)": float(row["Temperature (C)"]),
            "Temperature (K)": float(row["Temperature (C)"]) + 273.15,
            "Dwell Duration (h)": float(row["Dwell Duration (h)"])
        }
    # Save back to the JSON file
    with open(json_file_path, "w", encoding="utf-8") as jsonfile:
        json.dump(json_data, jsonfile, indent=4)
#', '.join(phases)

csv_info_to_json('Data/Dara-AIF_Evaluation2_Liam - Data.csv', 'Data/interpretations_llm_v1.json')

