# LLM XRD Interpretation Validation
## File Descriptions
- Llama_test.py: initial prompt formatting and information
- Llama_test2.py: updated prompt (Prompt used for Chat GPT)
- Llama_test3.py: currently updated prompt to correct missing and extra phases in Llama response
- steps_for_Liam.ipynb: file with process for creating final probabilities to compare LLMs
- csv_info_to_json.py: used to import all CSV synthesis information into the JSON file with the interpretation information
- Visualizations folder: used for all jupyter notebooks where graphs or charts were created
- Visualizations/visualize_ChatLlamaRwp.ipynb: currently used notebook for comparing LLMs
- results_check.ipynb: checks that that LLM response has the correct amount of phases per interpretation, any other checks I think of
- llm_prompt_template_(...) : all .txt files of this type are different versions of the txt file inserted into the prompt, _2 is the txt file used by Chat GPT
- The "Data" folder has all data for each Sample and LLM run, it is organized by prompt iteration, reference the "relevant files in data" doc for currently relevant data
