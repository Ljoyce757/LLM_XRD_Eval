# === To Read JSON Data and Print Specific Response ===
import json

with open("Data/llm_prompt_v3_response1.json", 'r') as f:
    data = json.load(f)

# Print out the prompt for all runs
#for run in data:    
#print(f"""{run}:\n {data[run]["Extracted_Dict"]}""")  # Print first 400 characters of the prompt

# Print the response for TRI_181
#print(f"TRI_181:\n {data['TRI_181']['Prompt']}")  # Print one prompt
#print(f"TRI_181:\n {data['TRI_181']['Extracted_Dict']}")  # Print one prompt response dictionary
print(data['TRI_87']['Prompt']) 