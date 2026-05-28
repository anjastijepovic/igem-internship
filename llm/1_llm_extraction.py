# This script performs LLM-based extraction of structured attribution data from raw text for multiple iGEM teams.

import os
import json
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm # For progress bars


# Configuration


MODEL_PATH = os.path.join(os.environ["DSDIR"], "HuggingFace_Models/meta-llama/Meta-Llama-3-70B-Instruct")
BATCH_SIZE = 4
NUM_RUNS_PER_PROMPT = 1 # 1 for single-round extraction logic
OUTPUT_FILE = "../../data/attributions/2022_attributions/llm_extraction_outputs/14_teams_jean_zay_results.json"

# Data Loading Helper Functions 


def load_attributions_json(filepath):
   """Loads a JSON file."""
   with open(filepath, "r", encoding="utf-8") as f:
       data = json.load(f)
   return data


def get_team_attributions_text(data, team_name):
   """Finds the raw attribution text for a given team."""
   for entry in data:
       if entry["teamName"] == team_name:
           return entry["attributionsText"]
   return None


def get_team_roster(roster_dict, team_name):
   """Retrieves the list of members for a team."""
   return roster_dict.get(team_name, [])


#  Data Null Normalization Helper Function (convert empty/none/null strings to None Python object)


def normalize_null_values(data):
    # Strings to treat as null
    null_strings = {"None", "none", "Null", "null", "NULL", "", " "}

    if isinstance(data, dict):
        # Recursively process dictionary values
        return {k: normalize_null_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        # Recursively process list elements
        return [normalize_null_values(item) for item in data]
    elif isinstance(data, str) and data.strip() in null_strings:
        # Convert null strings to Python None
        return None
    else:
        # Return all other types as is
        return data
    

# Cleaning Output Helper Function


def clean_output(raw_output):
   """Removes markdown code block delimiters from the raw LLM output."""
   return re.sub(r"^```[a-z]*\n|\n```$", "", raw_output.strip(), flags=re.MULTILINE)


# Prompt Logic 


def build_prompt_single(roster_name, team_raw_attributions):
   """
   Constructs the detailed prompt for the LLM extraction task.
   """
   example = '''
Example RosterName: Bb Smith


Example Raw Attribution Text:


"iGEM Project Description Team Members Alicé Johnson Team Leader Wet Lab Dry Lab Finances Alicé was responsible for labs, especially research, design of the SynBio solution, and modeling. She also helped with the team's finances and managed communication with various sponsors.\n Bob Smith Communications Bob worked in Communications, writing and producing the promotion video, and networking with other iGEM teams. All team members were a part of the project brainstorming."


Example Output:
{
"RosterName": "Bb Smith",
"RawTextName": "Bob Smith",
"TasksDescription": "Communications Bob worked in Communications, writing and producing the promotion video, and networking with other iGEM teams. All team members were a part of the project brainstorming."
}
'''
   return f"""
You are an AI that extracts structured JSON from text according to the following rules.


Your Task:
From the given raw attributions text, find all references to the given person and return a single JSON object with:
- RosterName: The exact roster name provided below (always included)
- RawTextName: The matched name from the raw text (or null if not mentioned)
- TasksDescription: Extracted phrases or sentences from the text that describe tasks that this person did in the team (or null if none found)


Rules:
- Match name approximately (account for spelling differences and cases where only first or last name is mentioned)
- Do NOT invent tasks, use only what is present in the text
- If multiple references to the person exist, merge all task descriptions into a single string
- If a sentence in the raw text applies to multiple people (e.g., “all team members did wiki coding”), include that sentence for the person (if it relates to them as well)
- Ignore roles or titles, focus on the person's task contributions
- If the person is not mentioned, keep RawTextName and TasksDescription as null


RosterName to look for:
{roster_name}


Raw Attribution Text:
\"\"\"{team_raw_attributions}\"\"\"


Example Input and Output:
{example}


Expected Output Format:
{{
"RosterName": "<roster name>",
"RawTextName": "<matched name from text or null>",
"TasksDescription": "<relevant phrases/sentences or null>"
}}


Return only a valid JSON object, nothing else.
"""


# Model Initialization 


def load_model_and_tokenizer():
   """Initializes the tokenizer and model for batch processing."""
   print(f"Loading model from {MODEL_PATH}...")
   tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
  
   # Important for batch processing
   tokenizer.pad_token = tokenizer.eos_token
   tokenizer.padding_side = "left"

   model = AutoModelForCausalLM.from_pretrained(
       MODEL_PATH,
       dtype=torch.bfloat16,
       device_map="auto", # Automatically spreads across the 4 GPUs
   )
   return tokenizer, model


# Batch Processing Logic


def run_inference_batch(tokenizer, model, prompts):
   """
   Takes a list of raw prompt strings, applies chat template, and runs batch inference.
   """
  
   # 1. Format prompts using the Llama chat template
   formatted_prompts = []
   for p in prompts:
       messages = [{"role": "user", "content": p}]
       # apply_chat_template converts the list of dicts to the string Llama expects
       formatted_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))


   # 2. Tokenize
   inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)


   # 3. Generate
   with torch.no_grad():
       output_ids = model.generate(
           **inputs,
           # Set big max tokens num to ensure long TasksDescription fields are fully generated and the JSON object is properly closed
           max_new_tokens=5000,
           do_sample=False, # Temperature 0 equivalent
           eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
       )


   # 4. Decode (strip the input prompt from the output to get just the response)
   generated_texts = []
   input_len = inputs.input_ids.shape[1]
  
   for i, out_ids in enumerate(output_ids):
       # Slice off the input tokens
       generated_text = tokenizer.decode(out_ids[input_len:], skip_special_tokens=True)
       generated_texts.append(generated_text)
      
   return generated_texts


# Main Execution 


def main():
   # 1. Load Data
   print("Loading datasets...")
   df_roster = pd.read_table("../../data/attributions/2022_attributions/team_rosters_2022.tsv")
   roster_2022_dict = df_roster.groupby("Team")["FullName"].apply(list).to_dict()
  
  # Selecting teams that exist in the survey but are not the evaluation or validation teams (we don't prompt the model for those, we will append our manual annotations later)
#    raw_attributions_data = load_attributions_json("../../data/attributions/2022_attributions/raw_attributions_2022.json") #this is for all attributions, but we only need the surveyed teams
   raw_attributions_data = load_attributions_json("../../data/attributions/2022_attributions/surveyed_teams_raw_attributions_2022.json")
   
   evaluation_teams_attributions = load_attributions_json("../../data/attributions/2022_attributions/manual_annotation/14_test_teams.json")
   evaluation_teams = [list(team_dict.keys())[0] for team_dict in evaluation_teams_attributions]
   
   validation_teams_attributions = load_attributions_json("../../data/attributions/2022_attributions/manual_annotation/5_validation_teams.json")
   validation_teams = [list(team_dict.keys())[0] for team_dict in validation_teams_attributions]

   surveyed_teams = [team_dict['teamName'] for team_dict in raw_attributions_data]
   
   surveyed_teams_without_manually_annotated_teams = [
    team for team in surveyed_teams
    if team not in evaluation_teams and team not in validation_teams
    ]

   # 2. Initialize Model
   tokenizer, model = load_model_and_tokenizer()

   # 3. Flatten workload for batching
   # Create a list of "Tasks" containing all info needed to reconstruct the result later
   work_queue = []
  
   for team_name in surveyed_teams_without_manually_annotated_teams: #use here evaluation_teams if you want to run only for the 14 evaluation teams
       team_raw_attributions = get_team_attributions_text(raw_attributions_data, team_name)
       if not team_raw_attributions:
           print(f"Skipping {team_name} (no text)")
           continue
          
       team_members = get_team_roster(roster_2022_dict, team_name)
      
       for member in team_members:
           # Only runs once per member now (NUM_RUNS_PER_PROMPT = 1)
           for run_idx in range(NUM_RUNS_PER_PROMPT):
               prompt_text = build_prompt_single(member, team_raw_attributions)
               work_queue.append({
                   "team": team_name,
                   "member": member,
                   "run_index": run_idx,
                   "prompt": prompt_text
               })

   print(f"Total prompts to process: {len(work_queue)}")


   # 4. Process in Batches
   results_buffer = [] # Store raw results
  
   for i in tqdm(range(0, len(work_queue), BATCH_SIZE), desc="Batch Processing"):
       batch_tasks = work_queue[i : i + BATCH_SIZE]
       batch_prompts = [t["prompt"] for t in batch_tasks]
      
       # Run Inference
       batch_responses = run_inference_batch(tokenizer, model, batch_prompts)
      
       # Parse and Store
       for task, raw_response in zip(batch_tasks, batch_responses):
           cleaned = clean_output(raw_response)
           parsed = None
           try:
               parsed = json.loads(cleaned)
               if isinstance(parsed, dict):
                   parsed = [parsed] # Normalize to list
           except json.JSONDecodeError:
               # Note the full raw output when an error occurs to aid in debugging
               print(f"JSON Error for {task['member']} in {task['team']}. Raw output:\n{cleaned[:100]}...")
               parsed = None
          
           # Store result linked to metadata
           results_buffer.append({
               "team": task["team"],
               "member": task["member"],
               "run_index": task["run_index"],
               "data": parsed
           })


   # 5. Reconstruct Output Structure
   # Transform the flat buffer back into the dictionary structure:
   # [{TeamName: {Round1: [data]}}]
  
   final_output_structure = []
  
   # Group by team
   teams_processed = list(set(item['team'] for item in results_buffer))
  
   for team in teams_processed:
       # Dynamically create the round keys based on NUM_RUNS_PER_PROMPT (which is 1)
       team_rounds = {f"Round{r+1}": [] for r in range(NUM_RUNS_PER_PROMPT)}
      
       # Filter items for this team
       team_items = [x for x in results_buffer if x['team'] == team]
      
       for item in team_items:
           r_key = f"Round{item['run_index']+1}"
           # Apply Null normalization
           normalized_data = normalize_null_values(item['data'])
           if normalized_data:
               if isinstance(normalized_data, list):
                   team_rounds[r_key].extend(normalized_data)
               else:
                   team_rounds[r_key].append(normalized_data)
                  
       final_output_structure.append({team: team_rounds})

   # 6. Save
   print(f"Saving results to {OUTPUT_FILE}")
   os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
   with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
       json.dump(final_output_structure, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
   main()