# Same as 1_llm_extraction.py but added username as secondary identifier for matching - this is the final py version that was used for text teams

import os
import json
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm 


# Configuration

MODEL_PATH = os.path.join(os.environ["DSDIR"], "HuggingFace_Models/meta-llama/Meta-Llama-3-70B-Instruct")
BATCH_SIZE = 4
NUM_RUNS_PER_PROMPT = 1 # can set to more rounds
OUTPUT_FILE = "../../data/attributions/2022_attributions/llm_json_outputs/Groningen.json"


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
   """Retrieves the list of members for a team. Now returns a list of dictionaries."""
   return roster_dict.get(team_name, [])


#  Data Null Normalization Helper Function (convert empty/none/null strings to None Python object)

def normalize_null_values(data):

    # Strings to treat as null 
    null_strings = {"none", "null", "n/a", "na", "", " "}

    if isinstance(data, dict):
        # Recursively process dictionary values
        return {k: normalize_null_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        # Recursively process list elements
        return [normalize_null_values(item) for item in data]
    elif isinstance(data, str):
        # Check if the stripped, lowercased string matches any null representation
        if data.strip().lower() in null_strings:
            return None
        # Check if the string only contained spaces (which results in "" after stripping)
        if data.strip() == "" and "" in null_strings:
             return None
        # Return all other strings as is
        return data
    else:
        # Return all other types (int, float, None, etc.) as is
        return data
    

# Cleaning Output Helper Function

def clean_output(raw_output):
    # Remove markdown code if present
    cleaned = re.sub(r"^```[a-z]*\n|\n```$", "", raw_output.strip(), flags=re.MULTILINE)

    def clean_tasks(match):
        value = match.group(1)

        # Replace problematic quotes and backslashes only inside the value 
        # Because sometimes there are quotes or backslashes in the attributions text that cause JSONDecode error

        value = value.replace('\"', "")
        value = value.replace("\'", "")   
        value = value.replace('"', "'")
        value = value.replace('\n', '\\n')
        value = value.replace("\\", "'")

        return f'"TasksDescription": "{value}"'

    # Replace only the TasksDescription value
    cleaned = re.sub(
        r'"TasksDescription"\s*:\s*"(.+?)"\s*(?=\n\s*})',
        clean_tasks,
        cleaned,
        flags=re.DOTALL
    )

    return cleaned

# The cleaned function sometimes does not work (there are still leftover backslashes) so added another one to run on the cleaned json:
def extract_first_json_object(s: str) -> str:
    # Grab from first { to last } 
    a = s.find("{")
    b = s.rfind("}")
    return s[a:b+1] if a != -1 and b != -1 and b > a else s


def repair_llm_json(s: str) -> str:
    s = extract_first_json_object(s)

    # Fix invalid JSON escapes like \' or \S etc:
    s = s.replace("\\'", "'")

    # Escape backslashes ONLY when they would create invalid escape sequences
    s = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)

    return s


# Prompt Logic - text prompt

def build_prompt_single(roster_name, username, team_raw_attributions):
   """
   Constructs the detailed prompt for the LLM extraction task, including the username
   as a secondary identifier for matching.
   """
   example = '''
Example RosterName (Primary): Bb Smith
Example Username (Secondary): Bsmith123

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
- Match name approximately (account for spelling differences and cases where only first or last name is mentioned).
- Prioritize matching the person using the RosterName: first search for the full name, and if no match is found, then search by first or last name only.
- If more than one roster member has the same first name, prioritize full first + last name matches. Two individuals should NOT have the exactly same TaskDescription. If you are not sure which attribution belongs to whom, leave TaskDescription empty.
- Process the entire raw text before assigning attributions; do not assign a first-name-only mention if a full first + last name variation appears elsewhere in the text.
- If the RosterName is not found, attempt to match using the Username as an alternative name mentioned in the raw text.
- Do NOT invent tasks, use only what is present in the text
- If multiple references to the person exist, merge all task descriptions into a single string
- If a sentence in the raw text applies to multiple people (e.g., “all team members did wiki coding”), include that sentence for the person (if it relates to them as well)
- Ignore roles or titles, focus on the person's task contributions
- If the person is not mentioned by either the RosterName or the Username, keep RawTextName and TasksDescription as null

RosterName (Primary Identifier) to look for:
{roster_name}

Username (Secondary Identifier) to look for:
{username}

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

# Prompt Logic - HTML prompt - need to add chunking logic because HTML prompts are too long
# Could me prompted only with table elements and text that exists outside can be extracted with text prompt

# def build_html_prompt_single(roster_name, username, team_raw_attributions):
#     example = '''
# Example RosterName (Primary): Bb Smith
# Example Username (Secondary): Bsmith123

# Example HTML Markup:

# "
# <html lang="en">
#   <head>
#     <meta charset="utf-8" />
#     <title>Blah Blah</title>
#   </head>
#   <body>
#     <section>
#       <pre>"iGEM Project Description Team Members Alicé Johnson Team Leader Wet Lab Dry Lab Finances Alicé was responsible for labs, especially research, design of the SynBio solution, and modeling. She also helped with the team's finances and managed communication with various sponsors.\n Bob Smith Communications Bob worked in Communications, writing and producing the promotion video, and networking with other iGEM teams. All team members were a part of the project brainstorming."</pre>
#     </section>
#   </body>
# </html>
# "

# Example Output:
# {
#   "RosterName": "B�b Smith",
#   "RawTextName": "Bob Smith",
#   "TasksDescription": "Communications Bob worked in Communications, writing and producing the promotion video, and networking with other iGEM teams. All team members were a part of the project brainstorming."
# }
# '''

#     return f"""
# You are an AI that extracts structured JSON from HTML markup according to the following rules.

# Your Task:
# From the given raw HTML markup with attributions, find all references to the given person and return a single JSON object with:
# - RosterName: The exact roster name provided below (always included)
# - RawTextName: The matched name from the HTML markup (or null if not mentioned)
# - TasksDescription: Extracted TEXT (phrases or sentences) from the HTML markup that describe tasks that this person did in the team (or null if none found)

# Rules:
# - Match name approximately (account for spelling differences and cases where only first or last name is mentioned).
# - Prioritize matching the person using the RosterName: first search for the full name, and if no match is found, then search by first or last name only.
# - If more than one roster member has the same first name, prioritize full first + last name matches. Two individuals should NOT have the exactly same TaskDescription. If you are not sure which attribution belongs to whom, leave TaskDescription empty.
# - Process the entire raw text before assigning attributions; do not assign a first-name-only mention if a full first + last name variation appears elsewhere in the text.
# - If the RosterName is not found, attempt to match using the Username as an alternative name mentioned in the raw text.
# - Do NOT invent tasks, use only what is present in the HTML markup
# - If multiple references to the person exist, merge all task descriptions into a single string
# - If a sentence in the HTML markup applies to multiple people (e.g., “all team members did wiki coding”), include that sentence for the person (if it relates to them as well)
# - Ignore roles or titles, focus on the person's task contributions
# - If the person is not mentioned by either the RosterName or the Username, keep RawTextName and TasksDescription as null

# RosterName (Primary Identifier) to look for:
# {roster_name}

# Username (Secondary Identifier) to look for:
# {username}

# Raw HTML Markup:
# \"\"\"{team_raw_attributions}\"\"\"

# Example Input and Output:
# {example}

# Expected Output Format:
# {{
#   "RosterName": "<roster name>",
#   "RawTextName": "<matched name from text or null>",
#   "TasksDescription": "<relevant phrases/sentences or null>"
# }}

# Return only a valid JSON object, nothing else.
# """


# Model Initialization 

def load_model_and_tokenizer():
   """Initializes the tokenizer and model for batch processing."""
   print(f"Loading model from {MODEL_PATH}...")
   tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
  
   # Crucial for batch processing
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
  
   # 1. Format prompts using the Llama 3 chat template
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
           # Set big max tokens num to ensure long TasksDescription fields are fully generated and the JSON object is properly closed - change if needed
           max_new_tokens=4000,
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
   # Group by Team, and aggregate FullName and Username into records (list of dicts) for easy iteration
   roster_2022_dict = df_roster.groupby("Team")[["FullName", "Username"]].apply(
       lambda x: x.to_dict('records')
   ).to_dict()

  # Selecting teams that exist in the survey but are not the evaluation or validation teams (we don't prompt the model for those, we will append our manual annotations later)
   raw_attributions_data = load_attributions_json("../../data/attributions/2022_attributions/surveyed_teams_raw_attributions_2022.json") # for text teams or evaluation teams
   raw_html_data = load_attributions_json("../../data/attributions/2022_attributions/html_attributions_2022_second_version.json") # for html teams
   
   # Uncomment if you want to use evaluation or validation teams in the prompting:
   
#    evaluation_teams_attributions = load_attributions_json("../../data/attributions/2022_attributions/manual_annotation/14_test_teams.json")
#    evaluation_teams = [list(team_dict.keys())[0] for team_dict in evaluation_teams_attributions]
   
#    validation_teams_attributions = load_attributions_json("../../data/attributions/2022_attributions/manual_annotation/5_validation_teams.json")
#    validation_teams = [list(team_dict.keys())[0] for team_dict in validation_teams_attributions]

   attributions_structures = pd.read_csv("../../data/attributions/2022_attributions/manual_annotation/wiki_attributions_structures_2022_teams.tsv", sep="\t")
   html_teams = attributions_structures['Team'][attributions_structures['Should be prompted with html'] == "yes"]
   text_teams = attributions_structures['Team'][attributions_structures['Should be prompted with text'] == "yes"]

   teams = ['Groningen'] # some extras

   # 2. Initialize Model
   tokenizer, model = load_model_and_tokenizer()


   # 3. Flatten workload for batching
   # Create a list of "Tasks" containing all info needed to reconstruct the result later
   work_queue = []
  
   for team_name in teams: # change if you want, for example, to use the evaluation_teams only
       team_raw_attributions = get_team_attributions_text(raw_attributions_data, team_name) #change to raw_attributions_data for text or eval teams, raw_html_data for html ones
       if not team_raw_attributions:
           print(f"Skipping {team_name} (no text)")
           continue
          
       # team_members_data now contains a list of dictionaries: [{'FullName': '...', 'Username': '...'}, ...]
       team_members_data = get_team_roster(roster_2022_dict, team_name)
      
       for member_data in team_members_data:
           member_fullname = member_data["FullName"]
           member_username = member_data["Username"]
           
           # Only runs once per member now (NUM_RUNS_PER_PROMPT = 1)
           for run_idx in range(NUM_RUNS_PER_PROMPT):
               # Pass both FullName and Username to the prompt builder
               prompt_text = build_prompt_single(member_fullname, member_username, team_raw_attributions) # change the building prompt function based on text or html
               work_queue.append({
                   "team": team_name,
                   "member": member_fullname,
                   "username": member_username, 
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
                repaired = repair_llm_json(cleaned)
                try:
                    parsed = json.loads(repaired)
                    if isinstance(parsed, dict):
                        parsed = [parsed]
                except:
                    # Note the full raw output when an error occurs to aid in debugging
                    print(f"JSON Error for {task['member']} in {task['team']}. Raw output:\n{cleaned}\n{'='*50}")
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
           
           # Apply null normalization
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