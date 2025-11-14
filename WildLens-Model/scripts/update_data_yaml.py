import yaml
from pathlib import Path
import sys


def update_yaml():
    """
    Automatically scans the 'data' directory for species folders
    and regenerates the 'data.yaml' file, matching the exact format
    including header comments and the final nc/names block.
    """
    try:
        # --- 1. Determine Paths ---
        SCRIPT_DIR = Path(__file__).resolve().parent
        ROOT_DIR = SCRIPT_DIR.parent
        DATA_DIR = ROOT_DIR / 'data'
        YAML_PATH = DATA_DIR / 'data.yaml'

        if not DATA_DIR.exists():
            print(f"ERROR: 'data' directory not found at: {DATA_DIR}")
            print("Please ensure this script is in a subdirectory of 'WildLens-Model' (e.g., 'scripts').")
            sys.exit(1)

        print(f"Scanning directory: {DATA_DIR}")
        print(f"Target file to update: {YAML_PATH}\n")

        # --- 2. Scan directories and build lists ---
        train_paths = []
        val_paths = []
        test_paths = []
        class_names = []

        for item in sorted(DATA_DIR.iterdir()):
            if not item.is_dir():
                continue

            species_name = item.name
            class_names.append(species_name)

            # Check for train/valid/test subdirs
            train_img_dir = item / 'train' / 'images'
            if train_img_dir.exists():
                train_paths.append((train_img_dir.relative_to(DATA_DIR)).as_posix())

            val_img_dir = item / 'valid' / 'images'
            if val_img_dir.exists():
                val_paths.append((val_img_dir.relative_to(DATA_DIR)).as_posix())

            test_img_dir = item / 'test' / 'images'
            if test_img_dir.exists():
                test_paths.append((test_img_dir.relative_to(DATA_DIR)).as_posix())

        if not class_names:
            print(f"ERROR: No species subdirectories found in {DATA_DIR}")
            sys.exit(2)

        print(f"Found {len(class_names)} classes (species).")
        print(f"Found {len(train_paths)} 'train' paths.")
        print(f"Found {len(val_paths)} 'val' paths.")
        print(f"Found {len(test_paths)} 'test' paths.")

        # --- 3. Load existing data (to preserve other keys) ---
        existing_data = {}
        if YAML_PATH.exists():
            try:
                with open(YAML_PATH, 'r', encoding='utf-8') as f:
                    existing_data = yaml.safe_load(f) or {}
                print(f"\nRead existing file: {YAML_PATH}")
            except Exception as e:
                print(f"WARNING: Could not read old {YAML_PATH}, a new file will be created. Error: {e}")

        # Remove keys that we will explicitly manage
        existing_data.pop('train', None)
        existing_data.pop('val', None)
        existing_data.pop('test', None)
        existing_data.pop('nc', None)
        existing_data.pop('names', None)

        # Add the scanned paths back. 'existing_data' now contains only *extra* keys.
        # We will dump these *first*, then our lists.
        data_to_dump = existing_data.copy()  # Preserve extra keys
        data_to_dump['train'] = train_paths
        data_to_dump['val'] = val_paths
        data_to_dump['test'] = test_paths

        # --- 4. Write the YAML file in the required format ---
        with open(YAML_PATH, 'w', encoding='utf-8') as f:

            # --- Write Header Comments (Manual) ---
            f.write("# Unified dataset config for WildLens (centralized)\n")
            f.write("# Dataset is organized per-species under WildLens-Model/data/<Species>/{train,valid,test}/\n")
            f.write("# Ultralytics supports multiple paths per split; we aggregate all species splits below.\n\n")
            f.write("# Paths are relative to this YAML file's location (WildLens-Model/data/)\n")

            # --- Dump the main data (train, val, test, and any extra keys) ---
            # We dump this part using the YAML library
            yaml.dump(data_to_dump, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

            # --- Write Footer Comments and Class Info (Manual) ---
            f.write("\n# Note: Some species may not include a 'test' split in this repo dump.\n")

            nc = len(class_names)
            names_string = ", ".join(class_names)

            f.write(f"\n# {nc} classes used by WildLens\n")  # Comment is dynamic
            f.write(f"nc: {nc}\n")
            f.write(f"names: [{names_string}]\n")  # Flow-style list

        print(f"\nSuccessfully updated file: {YAML_PATH}")
        print("Format includes manual headers and the required nc/names block at the end.")
        print("-----------------------------")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == '__main__':
    update_yaml()