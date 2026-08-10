import json
import yaml  # Requires: pip install pyyaml
from pathlib import Path

def extract_yaml_value(filepath):
    """
    Loads a YAML file and drills down to the 6th nested level,
    dynamically handling the unknown key at level 4.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        try:
            # safe_load is recommended for YAML files to prevent code execution
            data = yaml.safe_load(file)
            
            # ---------------------------------------------------------
            # REPLACE THESE WITH YOUR ACTUAL KNOWN KEYS
            # ---------------------------------------------------------
            level3_data = data["_wandb"]["value"]["e"]
            
            # Handle the dynamic Level 4 key
            # We grab all keys at this level, and take the first (and only) one
            dynamic_level4_key = list(level3_data.keys())[0]
            
            # Continue drilling down through level 4 to get to 5 and 6
            target_value = level3_data[dynamic_level4_key]["gpu"]
            
            return target_value
            
        except KeyError as e:
            print(f"Warning: Known key {e} not found in {filepath}")
            return None
        except (IndexError, AttributeError):
            print(f"Warning: Could not find the dynamic Level 4 key in {filepath}")
            return None
        except yaml.YAMLError as e:
            print(f"Error parsing YAML in {filepath}: {e}")
            return None

def process_runs(base_folder, target_filename):
    results = {}
    base_path = Path(base_folder)

    for subdir in base_path.iterdir():
        if not subdir.is_dir():
            continue

        # Extract the ID from the directory name (e.g., ndzzszr3)
        dir_name = subdir.name
        run_id = dir_name.split("-")[-1]

        target_file_path = subdir / 'files' / target_filename

        if target_file_path.is_file():
            extracted_value = extract_yaml_value(target_file_path)
            
            if extracted_value is not None:
                results[run_id] = extracted_value
            else:
                print('no', run_id)

    return results

if __name__ == "__main__":
    TARGET_DIRECTORY = "/users/project1/pt01299/synt/wandb_logs/wandb"
    SPECIFIC_FILE_NAME = "config.yaml"
    
    final_data = process_runs(TARGET_DIRECTORY, SPECIFIC_FILE_NAME)
    with open('/users/project1/pt01299/synt/extr_gpus.json', 'w') as f:
        json.dump(final_data, f, indent=2)
    print('over 1')

    TARGET_DIRECTORY = "/users/project1/pt01299/synt/wandb_logs2/wandb"
    final_data = process_runs(TARGET_DIRECTORY, SPECIFIC_FILE_NAME)
    with open('/users/project1/pt01299/synt/extr_gpus_g.json', 'w') as f:
        json.dump(final_data, f, indent=2)
    print('over 2g')
