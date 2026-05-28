'''
In this Python script, we fetch HTML attributions from the iGEM 2022 wiki for a list of teams.
We use the requests_html library to render JavaScript content and extract the HTML attributions.
First, we try to fetch attributions for a few skipped teams in 11_filter_survey_teams_for_llm.ipynb and append them to an existing JSON file of fetched HTML attributions.
Then, we do this for all teams since we noticed that some HTMLs were not collected properly in 11_filter_survey_teams_for_llm.ipynb (they did not contain attributions text, just a call to a JavaScript function).
The results are saved in an NDJSON file so they can be appended line by line, since this code is much slower than the one in the notebook.
These results are then converted to a final JSON file, html_attributions_2022_second_version.json.
'''


from requests_html import HTMLSession
import json
import os
import pandas as pd
import time
import random
from pathlib import Path
import requests

BASE = "https://2022.igem.wiki"

def convert_team_name(name):
    s = name.strip().lower()
    s = s.replace("_", "-")    
    return s


# # EXAMPLE FOR ONE TEAM
# session = HTMLSession()
# r = session.get("https://2022.igem.wiki/washington/attributions/")
# r.html.render()  # executes JS
# print(r.html.html)


# # SKIPPED TEAMS IN 11_filter_survey_teams_for_llm.ipynb ONLY

# # Path to the existing JSON file

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "attributions", "2022_attributions", "html_attributions_2022.json")
# OUTPUT_PATH = os.path.normpath(OUTPUT_PATH)


# skipped_teams = ['CAU_China', 'Costa_Rica', 'SUSTech_EMB', 'TecCEM', 'Stanford']
# results = []

# # Fetch HTML attributions for the skipped teams using HTMLSession

# for team in skipped_teams:
#     converted_team_name = convert_team_name(team)
#     url = f"{BASE}/{converted_team_name}/attributions" 
#     session = HTMLSession()
#     r = session.get(url)
#     r.html.render() 
#     attributions_html = r.html.html
#     results.append({
#     "teamName": team,
#     "attributionsText": attributions_html,
#     })

# # Append the results to the existing JSON file

# if os.path.exists(OUTPUT_PATH):
#     with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
#         try:
#             existing_results = json.load(f)
#         except json.JSONDecodeError:
#             existing_results = []
# else:
#     existing_results = []

# existing_results.extend(results)

# with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
#     json.dump(existing_results, f, ensure_ascii=False, indent=2)

# print(f"Appended {len(results)} new entries. Total: {len(existing_results)}")


# ALL TEAMS

# raw_attributions_df = pd.read_json("../data/attributions/2022_attributions/raw_attributions_2022.json")
# validation_teams_df = pd.read_json("../data/attributions/2022_attributions/manual_annotation/5_validation_teams_raw_attributions.json")
# merged_raw_attributions = pd.concat([raw_attributions_df, validation_teams_df], ignore_index=True)

# # Make teams lowercase for the domain, keep dashes; convert spaces/underscores to dashes 

# teams = (
#     merged_raw_attributions["teamName"]
#     .dropna()
#     .astype(str)
#     .drop_duplicates()
#     .tolist()
# )

# OUT_NDJSON = "../data/attributions/2022_attributions/html_attributions_2022_second_version.ndjson"

# session = HTMLSession()
# Path(OUT_NDJSON).parent.mkdir(parents=True, exist_ok=True)

# def fetch_team(team):
#     converted_team_name = convert_team_name(team) 
#     url = f"{BASE}/{converted_team_name}/attributions"
#     try:
#         t0 = time.perf_counter()
#         r = session.get(url, headers={"User-Agent": "Mozilla/5.0"})
#         r.html.render(timeout=60, sleep=1.0, keep_page=False) 
#         html = r.html.html
#         r.close()
#         elapsed = time.perf_counter() - t0
#         print(f"Added {team} in {elapsed:.1f}s")
#         return {"teamName": team, "attributionsText": html}
#     except Exception as e:
#         print(f"Skipped {team}, error: {e}")
#         return {"teamName": team, "error": str(e)}

# # Append line-by-line to NDJSON
# with open(OUT_NDJSON, "a", encoding="utf-8") as f:
#     for i, team in enumerate(teams, 1):
#         rec = fetch_team(team)
#         f.write(json.dumps(rec, ensure_ascii=False) + "\n")
#         f.flush() 
#         time.sleep(random.uniform(0.8, 1.6))

# session.close()
# print(f"Done. Wrote NDJSON to: {OUT_NDJSON}")

# # Convert NDJSON to final JSON 
# rows = [json.loads(line) for line in open(OUT_NDJSON, encoding="utf-8")]
# with open("../data/attributions/2022_attributions/html_attributions_2022_second_version.json", "w", encoding="utf-8") as f:
#     json.dump(rows, f, ensure_ascii=False, indent=2)

'''
Washington team is skipped with an error: "requests_html.MaxRetries: Unable to render the page. Try increasing timeout".
We will fetch it separately here, with the code used in the notebook 11_filter_survey_teams_for_llm.ipynb, since it worked there.
We will then replace the Washington entry in the final JSON file, html_attributions_2022_second_version.json.
Also, the previous code fetches the 404 page of the team NJU-China that has a different attributions domain.
We will fetch it from the correct domain - https://2022.igem.wiki/nju-china/team/attribution/.
'''


team = "NJU-China"
converted_team_name = convert_team_name(team)
# url = f"{BASE}/{converted_team_name}/attributions"
url = f"{BASE}/{converted_team_name}/team/attribution/"

resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
new_entry = {
    "teamName": team,
    "attributionsText": resp.text,
}

json_path = "../data/attributions/2022_attributions/html_attributions_2022_second_version.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Replace the team entry completely 
for i, entry in enumerate(data):
    if entry.get("teamName") == team:
        data[i] = new_entry 
        break
else:
    data.append(new_entry)

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Entry updated successfully.")

