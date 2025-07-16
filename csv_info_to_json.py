import csv
import json
import ast
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

def remove_after_underscore(s):
    """
    Returns the part of the string before the first underscore.
    
    Example:
    remove_after_underscore("TI122025_61") -> "TI122025"
    """
    return s.split('_')[0]

def csv_info_to_json(csv_file_path, json_file_path):
    csv_data = read_csv_file(csv_file_path)
    json_data = read_json_file(json_file_path)

    for row in csv_data: #iterate over each row in the CSV file
        run_name = row["Name"].strip()
        run_name = run_name.replace("-","_")
        if run_name not in json_data:
            continue
        # Populate the JSON data with the relevant fields from the CSV
        temperature = row["Temperature (C)"]
        duration = row["Dwell Duration (h)"]
        precursors = ast.literal_eval(row["Precursors"])
        furnace = row["Furnace"]
        if len(row["Precursors"]) == 3 and all([temperature,duration,precursors,furnace]):
            json_data[run_name]["Synth_Conditions"] = {
                "Target": row["Target"],
                "Precursor 1": remove_after_underscore(precursors[0]),
                "Precursor 2": remove_after_underscore(precursors[1]),
                "Precursor 3": remove_after_underscore(precursors[2]),
                "Furnace": furnace,
                "Temperature (C)": float(temperature),
                "Temperature (K)": float(temperature) + 273.15,
                "Dwell Duration (h)": float(row["Dwell Duration (h)"])
            }
        elif all([temperature,duration,precursors,furnace]): 
            json_data[run_name]["Synth_Conditions"] = {
                "Target": row["Target"],
                "Precursor 1": remove_after_underscore(precursors[0]),
                "Precursor 2": remove_after_underscore(precursors[1]),
                "Furnace": furnace,
                "Temperature (C)": float(temperature),
                "Temperature (K)": float(temperature) + 273.15,
                "Dwell Duration (h)": float(row["Dwell Duration (h)"])
            }
    # Save back to the JSON file
    with open(json_file_path, "w", encoding="utf-8") as jsonfile:
        json.dump(json_data, jsonfile, indent=4)
#', '.join(phases)

csv_info_to_json('Data/synthesis_ARR.csv', 'Data/train_final_weights.json')
csv_info_to_json('Data/synthesis_PG.csv', 'Data/train_final_weights.json')
csv_info_to_json('Data/synthesis_TRI.csv', 'Data/train_final_weights.json')

