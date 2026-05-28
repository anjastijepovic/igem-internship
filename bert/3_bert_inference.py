
# This script performs inference using a trained BERT model to classify task descriptions from a JSON file of LLM outputs.
# For each member's task description, it predicts described tasks based on learned thresholds and saves the results to a TSV file.

# --------------------- IMPORTS ---------------------

import sys
sys.modules["deepspeed"] = None

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()


# --------------------- CONFIG ---------------------

TRAINED_MODEL_PATH = "../../data/attributions/2022_attributions/bert/bert_training_data/model_results/trained_model"
INPUT_JSON = "../../data/attributions/2022_attributions/all_surveyed_teams_2022_task_descriptions.json"
OUTPUT_TSV = "../../data/attributions/2022_attributions/bert/bert_inference_outputs/bert_classified_tasks_for_surveyed_teams_2022_without_manually_annotated_ones.tsv"
# INPUT_JSON = "../../data/attributions/2022_attributions/manual_annotation/14_test_teams.json" # for eveluation
# OUTPUT_TSV = "../../data/attributions/2022_attributions/bert/bert_inference_outputs/14_teams_bert_classified_tasks.tsv"
THRESHOLDS_PATH = "../../data/attributions/2022_attributions/bert/bert_training_data/model_results/best_thresholds.csv"
TASK_PROBABILITIES_OUTPUT_JSON = "../../data/attributions/2022_attributions/bert/bert_inference_outputs/bert_task_probabilities_for_surveyed_teams_2022_without_manually_annotated_ones.json"

BATCH_SIZE = 32
MAX_LENGTH = 512

# For this code to work, the order of task labels should be the same as the one made in bert training
TASK_LABELS = [
    "software","conceptualization","public engagement","writing","investigation",
    "hardware","project administration","entrepreneurship","fundraising",
    "background research","safety","analysis","visualization","data curation",
]


#  --------------------- JSON PARSING ---------------------

# Function to parse the IGEM JSON file into a DataFrame with columns: Team, FullName, TasksDescription
def parse_igem_json(json_path):
    with open(json_path, 'r', encoding='utf-8', errors='replace') as f:
        data = json.load(f)
    
    rows = []
    # Data is usually a list containing one or more team dictionaries
    for entry in data:
        for team_name, content in entry.items():
            # Standardize everything into a list of members
            members_list = []
            # Case 1: content is a dict with "Round1", "Round2", etc.
            if isinstance(content, dict):
                for round_key in content:
                    members_list.extend(content[round_key])
            # Case 2: content is a direct list of members
            elif isinstance(content, list):
                members_list = content

            for member in members_list:
                # Use .get() to avoid KeyErrors and default to empty string if null
                desc = member.get("TasksDescription", "")
                if desc is None: desc = "" 
                
                rows.append({
                    "Team": team_name,
                    "FullName": member.get("RosterName", ""), # FullName is the member name used in roster data. Use empty string if null
                    "TasksDescription": desc
                })
    return pd.DataFrame(rows)


# --------------------- DATASET CLASS ---------------------

class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        # We replace actual empty strings with a space just for the tokenizer to stay stable
        safe_texts = [t if t.strip() != "" else " " for t in texts]
        self.encodings = tokenizer(safe_texts, truncation=True, padding="max_length", max_length=max_len)
    def __len__(self): return len(self.encodings.input_ids)
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}


# --------------------- LOAD MODEL and TOKENIZER ---------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    TRAINED_MODEL_PATH, 
    attn_implementation="eager"
).to(device)
model.eval()

# Load and Parse
df = parse_igem_json(INPUT_JSON)

# Drop manually annotated teams (but keep Sogang_Korea and Aboa)
teams_to_exclude = ['Aalto-Helsinki','BostonU_HW','CPU_Nanjing','CSMU_Taiwan','Cambridge','Freiburg','Goettingen','ICT-Mumbai','Montpellier','TU_Braunschweig','Technion-Israel','UPNAvarra_Spain']

df = df[~df["Team"].isin(teams_to_exclude)].reset_index(drop=True)

# Create Dataset and Loader
dataset = InferenceDataset(df["TasksDescription"].tolist(), tokenizer, MAX_LENGTH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


# --------------------- INFERENCE ---------------------

all_probs = []
with torch.no_grad():
    for batch in tqdm(loader, desc="Classifying"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

probs_matrix = np.vstack(all_probs)


# --------------------- APPLY THRESHOLDS ---------------------

thresh_df = pd.read_csv(THRESHOLDS_PATH)
thresh_map = dict(zip(thresh_df['label'], thresh_df['best_threshold']))

# Align thresholds with TASK_LABELS order
thresholds = [thresh_map[label] for label in TASK_LABELS]

predicted_tasks = []
for i, row in df.iterrows():
    # Check if the description is empty, null, or just whitespace
    if not str(row["TasksDescription"]).strip() or pd.isna(row["TasksDescription"]):
        predicted_tasks.append(None) 
    else:
        # Get tasks where probability >= its specific threshold
        labels = [
            TASK_LABELS[j] 
            for j in range(len(TASK_LABELS)) 
            if probs_matrix[i, j] >= thresholds[j]
        ]
        predicted_tasks.append(labels)

df["Tasks"] = predicted_tasks


# --------------------- MAKE TASK PROBABILITIES JSON ---------------------

task_probabilities_json = {}                          

for i, row in df.iterrows():                           
    team = row["Team"]                               
    member = row["FullName"]                         

    if team not in task_probabilities_json:            
        task_probabilities_json[team] = {}            

    task_probabilities_json[team][member] = {         
        label: float(probs_matrix[i, j])               
        for j, label in enumerate(TASK_LABELS)     
    }     


# --------------------- SAVE RESULTS ---------------------

# Task probabilities JSON
os.makedirs(os.path.dirname(TASK_PROBABILITIES_OUTPUT_JSON), exist_ok=True)  
with open(TASK_PROBABILITIES_OUTPUT_JSON, "w", encoding="utf-8") as f:        
    json.dump(task_probabilities_json, f, indent=2, ensure_ascii=False)       

print(f"Saved task probabilities to {TASK_PROBABILITIES_OUTPUT_JSON}")   

# Classified tasks TSV
os.makedirs(os.path.dirname(OUTPUT_TSV), exist_ok=True)
df.to_csv(OUTPUT_TSV, sep="\t", index=False)
print(f"Inference complete. Saved to {OUTPUT_TSV}")