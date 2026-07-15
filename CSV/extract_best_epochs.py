import json
from pathlib import Path


KEY = 'eps_best'


def parse_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith(KEY):
                return line.split(KEY)[1].strip()
    return None


def extract_data_from_files(base_folder, target_filenames):
    results = {}
    base_path = Path(base_folder)

    for subdir in base_path.iterdir():
        if not subdir.is_dir():
            continue

        dir_key = subdir.name # id
        for target_filename in target_filenames: # one of ph1, ph2, ph3
            target_file_path = subdir / target_filename
            if target_file_path.is_file():
                try:
                    extracted_value = parse_file(target_file_path)
                    if extracted_value is not None:
                        results[dir_key] = extracted_value
                    else:
                        print(f"Notice: Target text not found in {target_file_path}")
                except Exception as e:
                    print(f"Error reading {target_file_path}: {e}")
                continue

    return results


if __name__ == "__main__":
    TARGET_DIRECTORY = "/users/project1/pt01299/synt/fts2"
    TARGET_DIRECTORY = "/users/project1/pt01299/synt/fts2g"
    SPECIFIC_FILE_NAMES = ["stats_ph1.txt", "stats_ph2.txt", "stats_ph3.txt"]
    
    final_data = extract_data_from_files(TARGET_DIRECTORY, SPECIFIC_FILE_NAMES)
    with open('/users/project1/pt01299/synt/comb_stats_g.json', 'w') as f:
        json.dump(final_data, f, indent=2)
