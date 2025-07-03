import openai
import os
import json
import textwrap
import ast
import ijson 
import tempfile
from dotenv import load_dotenv
import re
import time 
import glob
from pymatgen.core import Composition

load_dotenv()

# --- Start llama API ---
client = openai.OpenAI(
    api_key=os.getenv('CBORG_API_KEY'), # Please do not store your API key in the code
    base_url="https://api.cborg.lbl.gov" # Local clients can also use https://api-local.cborg.lbl.gov
)

# --- Sub Functions---
def load_prompt_template(file_path): # open txt file for the prompt template
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return ""
def remove_spaces(s):
    return s.replace(" ", "")

def type_of_furnace(furnace): #convert furnace name to a sentence rather than label (eg. 'BF' -> 'Box furnace with ambient air')
    furnace = remove_spaces(furnace)
    furnace = furnace.strip()  # Remove spaces from the furnace name
    if furnace == 'BF':
        furnace = 'Box furnace with ambient air'
    elif furnace == 'TF-Ar':
        furnace = 'Tube furnace with flowing Argon (flow rate unknown)'
    elif furnace == 'TF-Ar+H2':
        furnace = 'Tube furnace with flowing Argon and Hydrogen (flow rate unknown)'
    elif furnace == 'TF-O2':
        furnace = 'Tube furnace with flowing Oxygen (flow rate unknown)'
    return furnace

def load_json(file_path): # Load JSON Data
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("JSON decode error — file is empty or malformed.")
                return {}
    else:
        return {}  # Return empty dict if file doesn't exist


def update_json(data, key, value): # Modify/Add Data
    data[key] = value
    return data

def save_json(file_path, data): # Save JSON Data
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def split_phase_and_spacegroup(phase_str):
    """
    Splits a phase string like 'VO2_14' into phase name and space group.
    Returns (phase_name, space_group) as strings.
    """
    if "_" in phase_str:
        phase_name, space_group = phase_str.rsplit("_", 1)
        return phase_name, space_group
    else:
        return phase_str, None
def interpret_dict_list(run_interpretations): 
    """
    interprets a dictionary and returns a new dictionary 
    with lists of phases for each interpretation
    """
    interpret_dict = {}  # Dictionary to hold interpretations and their phases in correct format
    for interpret in run_interpretations: # iterate over each interpretation in the run
        if interpret.startswith("I_"):  # Check if the key starts with "I_"
            # Extract the interpretation number and phases
            interpret_dict[interpret] = []  # Initialize an empty list for this interpretation
            for i in range(len(run_interpretations[interpret]["phases"])):
                phase_str = run_interpretations[interpret]["phases"][i] # e.g., "ZrTiO4_18"
                phase_name, space_group = split_phase_and_spacegroup(phase_str) #splits string into phase name and space group 
                wf = round(run_interpretations[interpret]["weight_fraction"][i],2) # get the weight fraction and round to 2 decimal places
                detailed_info = describe_clean_composition(phase_name)
                written_phase = (f"{phase_name} (space group {space_group}, weight fraction {round(wf, 2)}%, "
                            f"{detailed_info.split(', ', 1)[1]})")
                interpret_dict[interpret].append(written_phase)
    return interpret_dict
def comp_bal_score(run_interpretations):
    comp_bal_dict = {}
    for interpret in run_interpretations:
        if interpret.startswith("I_"):
            comp_bal_dict[interpret] = run_interpretations[interpret]["balance_score"]
    return comp_bal_dict

def phase_list(run_conditions):
    if "Precursor 3" not in run_conditions: 
    # If there are only two precursors, ignore the third
        phases = [run_conditions["Precursor 1"], run_conditions["Precursor 2"]]
    else:   
        # If there are three precursors, include all three
        phases = [run_conditions["Precursor 1"], run_conditions["Precursor 2"], run_conditions["Precursor 3"]]
    return phases

# def extract_dict_from_llm_output(llm_output, max_size_mb=10):
#     """
#     Extracts and parses a dictionary from a language model output string.
#     Handles markdown code blocks, malformed JSON, and large payloads.
#     """

#     def extract_dict_string(text):
#         # Try to extract JSON from code block
#         code_block = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL)
#         if code_block:
#             return code_block.group(1)

#         # Fall back to the largest brace block
#         curly_block = re.search(r"(\{.*\})", text, re.DOTALL)
#         if curly_block:
#             return curly_block.group(1)

#         return None

#     def is_large_payload(text, limit_mb):
#         size_mb = len(text.encode('utf-8')) / 1024 / 1024
#         return size_mb > limit_mb, size_mb

#     def try_json_parse(text):
#         try:
#             return json.loads(text)
#         except json.JSONDecodeError:
#             return None

#     def clean_json_string(text):
#         # Fix common LLM formatting issues
#         text = text.replace("'", '"')
#         text = re.sub(r",\s*([}\]])", r"\1", text)  # Remove trailing commas
#         text = re.sub(r'(?<!")(\b\w+\b)\s*:', r'"\1":', text)  # Quote unquoted keys
#         return text

#     dict_str = extract_dict_string(llm_output)
#     if not dict_str:
#         print("No dictionary-like structure found in the output.")
#         return None

#     # Handle large payloads with ijson
#     too_large, size_mb = is_large_payload(dict_str, max_size_mb)
#     if too_large:
#         print(f"Dictionary is large ({size_mb:.2f} MB). Using streaming parser.")
#         with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as tmpfile:
#             tmpfile.write(dict_str)
#             tmpfile.flush()
#             tmpfile.seek(0)
#             result = {}
#             for key, value in ijson.kvitems(tmpfile, ''):
#                 result[key] = value
#         return result

#     # Try standard JSON parse
#     parsed = try_json_parse(dict_str)
#     if parsed is not None:
#         return parsed

#     # Try again after cleaning
#     cleaned = clean_json_string(dict_str)
#     parsed = try_json_parse(cleaned)
#     if parsed is not None:
#         return parsed

#     # Last resort: literal_eval
#     try:
#         return ast.literal_eval(dict_str)
#     except Exception as e:
#         print(f"Failed to parse dictionary using literal_eval: {e}")
#         return None
import re
import json
import ast
import tempfile
import ijson

def extract_dict_from_llm_output(llm_output, max_size_mb=10):
    """
    Extracts and parses a dictionary from LLM output.
    Handles markdown code blocks, malformed JSON, large payloads, and nested braces.
    """

    def extract_dict_string(text):
        # 1. Try to extract JSON from markdown code block
        code_block = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL)
        if code_block:
            return code_block.group(1)

        # 2. Try balanced brace extraction
        return extract_balanced_braces(text)

    def extract_balanced_braces(text):
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        return None

    def is_large_payload(text, limit_mb):
        size_mb = len(text.encode('utf-8')) / 1024 / 1024
        return size_mb > limit_mb, size_mb

    def try_json_parse(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"JSON decode failed: {e}")
            return None

    def clean_json_string(text):
        # 1. Replace single quotes with double quotes
        text = text.replace("'", '"')

        # 2. Remove trailing commas inside objects and arrays
        text = re.sub(r",\s*([}\]])", r"\1", text)

        # 3. Quote unquoted keys (e.g., V2O3: → "V2O3":)
        text = re.sub(r'(?<!")(\b[\w\s\-/]+\b)\s*:', r'"\1":', text)

        return text

    dict_str = extract_dict_string(llm_output)
    if not dict_str:
        print("No dictionary-like structure found in the output.")
        return None

    # print("Raw extracted string (first 500 chars):")
    # print(dict_str[:500])

    too_large, size_mb = is_large_payload(dict_str, max_size_mb)
    if too_large:
        print(f"Payload too large ({size_mb:.2f} MB). Using ijson.")
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as tmpfile:
            tmpfile.write(dict_str)
            tmpfile.flush()
            tmpfile.seek(0)
            result = {}
            for key, value in ijson.kvitems(tmpfile, ''):
                result[key] = value
        return result

    # First attempt: raw
    parsed = try_json_parse(dict_str)
    if parsed is not None:
        return parsed

    # Second attempt: cleaned
    cleaned = clean_json_string(dict_str)
    # print("Trying cleaned version (first 500 chars):")
    # print(cleaned[:500])
    parsed = try_json_parse(cleaned)
    if parsed is not None:
        return parsed

    # Last resort: literal_eval
    try:
        return ast.literal_eval(dict_str)
    except Exception as e:
        print(f"literal_eval failed: {e}")
        return None


def put_response_in_json(extracted_dict, json_file,run_name,save_json_file):
    if extracted_dict is not None:
        # Update the JSON file with the new interpretation
        for interpret in extracted_dict:
            if interpret not in json_file[run_name] and interpret.startswith("I_"):
                # This will hopefully catch any false interpreations where LLM was not given interpretation info
                print(f"Interpretation name not in JSON file: {interpret}")
                continue #skip this interpretation
            LLM_phases_likelihood_llama = extracted_dict[interpret].get("Likelihoods", {})
            LLM_phases_explanation_llama = extracted_dict[interpret].get("Explanations", {})
            LLM_interpretation_likelihood_llama = extracted_dict[interpret].get("Interpretation_Likelihood", None)
            LLM_interpretation_explanation_llama = extracted_dict[interpret].get("Interpretation_Explanation", None)
            json_file[run_name][interpret]["LLM_phases_likelihood_llama"] = LLM_phases_likelihood_llama
            json_file[run_name][interpret]["LLM_phases_explanation_llama"] = LLM_phases_explanation_llama
            json_file[run_name][interpret]["LLM_interpretation_likelihood_llama"] = LLM_interpretation_likelihood_llama
            json_file[run_name][interpret]["LLM_interpretation_explanation_llama"] = LLM_interpretation_explanation_llama
        # Write the data to the new JSON file
        with open(save_json_file, "w", encoding="utf-8") as f:  
            json.dump(json_file, f, indent=4)
def store_prompt_response(run_name, llm_input_response, extracted_dict, prompt, content):
    if not os.path.exists(llm_input_response):
        # Create an empty JSON file
        with open(llm_input_response, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=4)
    response_dict = load_json(llm_input_response)
    if run_name not in response_dict:
        response_dict[run_name] = {}
    response_dict[run_name]["Prompt"] = prompt # add the prompt to the input response
    response_dict[run_name]["response"] = content # add the response content to the input response
    if extracted_dict is not None: # check if the dictionary was extracted successfully
        response_dict[run_name]["Extracted_Dict"] = extracted_dict
    else: 
        response_dict[run_name]["Extracted_Dict"] = "No dictionary found in response"
    save_json(llm_input_response, response_dict) #save to the json file

def describe_clean_composition(formula_str, digits=4, max_denominator=30):
    comp = Composition(formula_str)

    # Get rounded fractional composition
    frac_dict = {el: round(amt, digits) for el, amt in comp.fractional_composition.get_el_amt_dict().items()}

    # Reconstruct and quantize
    frac_comp = Composition(frac_dict)
    quantized_formula_str = frac_comp.get_integer_formula_and_factor(max_denominator=max_denominator)[0]
    quantized_dict = Composition(quantized_formula_str).get_el_amt_dict()

    # Format with clean chemical formula style
    ordered_elements = list(frac_dict.keys())
    quantized_str = ''.join(
        f"{el}" if int(quantized_dict[el]) == 1 else f"{el}{int(quantized_dict[el])}"
        for el in ordered_elements if el in quantized_dict
    )

    return f"{formula_str}, fractional_composition = {frac_dict}, approximately equal to {quantized_str}"

# === Full function ===
def Llama_response_oneRun(json_file, run_name,save_json_file, save_promptresponse, retries=1):
    """
    This function is used to run the Llama AI model on a specific run's data.
    It loads the JSON file, extracts the necessary information, and constructs a prompt for the model.
    """
    #get information from json file to fill in these conditions
    if run_name not in json_file:
        print("Sample not in json_file data set")
        return # avoids a key error if the run_name is not in the json_file
    run_interpretations = {k: v for k, v in json_file[run_name].items() if k.startswith("I_")}# all data from the run name in the json file
    run_conditions = json_file[run_name]["Synth_Conditions"] # synthesis conditions for the run

    # --- Data from the Specific Run---
    model = "Llama-4-Scout-17B-16E-Instruct"
    phases = phase_list(run_conditions)
    synth_conditions = 'gram-quantity precursors are mixed and heated in a furnace'  # e.g., "gram-quantity precursors are mixed and heated in a furnace."
    target_phase = run_conditions["Target"]  # e.g., "ZrTiO4"
    precursors = ', '.join(phases) # e.g., "ZrO2, TiO2"
    temp_k = run_conditions["Temperature (K)"]  # e.g., 1273.15 K
    temp_c = run_conditions["Temperature (C)"]  # e.g., 1000°C
    time_dwell = float(run_conditions["Dwell Duration (h)"])  # e.g., 4.0 hours
    furnace_type = type_of_furnace(run_conditions["Furnace"])  # e.g., "Box furnace with ambient air"

    # --- Interpretations and Composition Balance Scores ---

    all_phases = interpret_dict_list(run_interpretations)
    # a dictionary with "name" and "phases" categories:
    # names should correspond to _# for the interpretation number 
    # phases is a list of phases in format: ZrTiO4 (space group 18, weight fraction 80%)
    composition_balance_scores = comp_bal_score(run_interpretations)
    # dictionary with "name" and "score" categories 
    # name: the interpretation (same as all_phases)
    # score: the composition balance score for a given interpretation

    # --- Combine Into One Statement ---
    synthesis_data = f"Solid state synthesis: {synth_conditions}.\n    Target: {target_phase}\n    Precursors: {precursors}\n    Temperature: {temp_k} K ({temp_c}°C)\n    Dwell Duration: {time_dwell} hours\n    Furnace: {furnace_type}"
    # Solid state synthesis; gram-quantity precursors are mixed and heated in a furnace.
    # Target: ZrTiO4  
    # Precursors: ZrO2, TiO2  
    # Temperature: 1273.15 K (1000°C)  
    # Dwell Duration: 4.0 hours  
    # Furnace: Box furnace with ambient air


    # ---Prompt Information---
    prompt = textwrap.dedent(f"""\
    Given the following synthesis data:
    {synthesis_data}

    Below are multiple proposed phase interpretations. For each interpretation, determine the likelihood that the listed solid phases have formed under the given synthesis conditions.
    Take into account:
    - Whether the oxidation state is thermodynamically plausible (based on precursors, temperature, and synthesis atmosphere).
    - Whether the specific polymorph (space group) is known to be stable at the synthesis temperature and pressure. If multiple polymorphs exist for the same composition, prefer the polymorph known to be stable under the synthesis conditions.
    - Whether the overall elemental composition of the phases, weighted by their fractions, matches the expected target composition. Interpretations with large elemental imbalances (e.g., excess or missing cations) should be penalized. Use the provided composition balance score as an indicator of this match.

    """)

    # Add interpretation info
    prompt += "\nInterpretations:\n"
    for name, phases in all_phases.items():
        prompt += f"- {name}: {', '.join(phases)}\n"

    # Add composition balance scores
    prompt += "\nComposition balance scores:\n"
    for name, score in composition_balance_scores.items():
        prompt += f"- {name}: {round(score, 3)}\n"

    prompt += load_prompt_template("llm_prompt_template_2.txt")
    #print(prompt)
    #--- Prompt --- 
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in material synthesis and phase prediction. Use thermodynamics, kinetics, and polymorph knowledge to evaluate stability and likelihood of observed phases."},
                {"role": "user", "content": prompt},
                ],
                temperature=0,
                seed=42,
                stream=False,
                #max_tokens=5000
            )
        
        content = response.choices[0].message.content.strip() # Get the content from the response 
        
        extracted_dict = extract_dict_from_llm_output(content) # take out the dictionary from the response content
        if not extracted_dict and retries < 5: # if it cannot get the dictionary
            failed_dict = load_json("Data/prompt3/LLM_failedDictionary.json")
            if save_json_file not in failed_dict:
                failed_dict[save_json_file] = {}
            failed_dict[save_json_file][run_name] = content
            save_json("Data/prompt3/LLM_failedDictionary.json", failed_dict)
            Llama_response_oneRun(json_file, run_name, save_json_file, save_promptresponse,retries=retries+1) #recursively call the function again
            return # exit out of this run 

        #stores prompt and response information for every run (usefull for debugging)
        store_prompt_response(run_name, save_promptresponse, extracted_dict, prompt, content) 

        put_response_in_json(extracted_dict, json_file, run_name,save_json_file) # save the response to the json file 
        
    except Exception as e:
            print(f"Error calling model: {e}")
# --- End of the Big Function ---

# --- Main Execution for All Runs --- 
start_time = time.time()  # Start timer

json_file = load_json("Data/test_final_weights.json")

# === Find the next available file name ===
base = "Data/prompt3/interpretations_llm_v1_llama"
existing = glob.glob(f"{base}*.json")
nums = [int(re.search(r"llama(\d+)\.json", f).group(1)) for f in existing if re.search(r"llama(\d+)\.json", f)]
next_num = max(nums) + 1 if nums else 1
save_json_file = f"{base}{next_num}.json"  # File to save the results

# === Find the next available file name ===
base1 = "Data/prompt3/llm_prompt_v1_response"
# existing1 = glob.glob(f"{base1}*.json")
# nums1 = [int(re.search(r"response(\d+)\.json", f).group(1)) for f in existing1 if re.search(r"response(\d+)\.json", f)]
next_num1 = next_num
save_promptresponse = f"{base1}{next_num1}.json"  # File to save the results

os.makedirs(os.path.dirname(save_json_file), exist_ok=True)  # Ensure Data/ exists

for run in json_file:
    if "Synth_Conditions" in json_file[run]:
        has_interpretation = any(k.startswith("I_") for k in json_file[run].keys())
        if has_interpretation:
            run_name = run
            print(f"Running Llama response for: {run_name}")
            Llama_response_oneRun(json_file, run_name, save_json_file, save_promptresponse)
            
#Comment out 
#Llama_response_oneRun(json_file,"TRI_181",save_json_file, save_promptresponse) # Example run for debugging or running an individual sample 

end_time = time.time()  # End timer
elapsed = end_time - start_time
print(f"\nTotal run time: {elapsed:.2f} seconds")