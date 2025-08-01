# LLM XRD Interpretation Validation
## File Descriptions
- Llama_test.py: initial prompt formatting and information
- Llama_test2.py: Base prompt with approximate equal and other changes (Prompt used for Chat GPT)
- Llama_test3.py: Revised prompt with all changes to the base prompt (Llama_test2.py)
- AIF_Calculation.ipynb: file with process for creating final probabilities to compare LLMs
- csv_info_to_json.py: used to import all CSV synthesis information into the JSON file with the interpretation information
- Visualizations folder: used for all jupyter notebooks where graphs or charts were created
- Visualizations/visualize_FinalGraphics_DataAnalysis.ipynb: most of the graphs and data processed for the paper
- Visualizations/visualize_useful_graphics.ipynb: most of the suplementary or useful graphics and data analysis
- results_check.ipynb: checks that that LLM response has the correct amount of phases per interpretation
- llm_prompt_template_(...) : all .txt files of this type are different versions of the txt file inserted into the prompt, _2 is the txt file used by Chat GPT
