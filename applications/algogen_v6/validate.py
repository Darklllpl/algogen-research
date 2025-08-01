# validate.py
import json
from pathlib import Path
import jsonschema

def validate_svl_file(json_path, schema_path):
    """Validate if an SVL JSON file conforms to specification."""
    print(f"--- Validating: {json_path.name} ---")
    try:
        # Load schema and data
        schema = json.loads(Path(schema_path).read_text(encoding='utf-8'))
        data = json.loads(Path(json_path).read_text(encoding='utf-8'))

        # Execute validation
        jsonschema.validate(instance=data, schema=schema)

        print(f"[Success] File {json_path.name} conforms to SVL 5.0 specification.")
        return True
    except jsonschema.exceptions.ValidationError as e:
        print(f"[Failed] File {json_path.name} failed validation.")
        print(f"Error: {e.message}")
        print(f"Error path: {list(e.path)}")
        return False
    except FileNotFoundError:
        print(f"[Failed] File not found: {json_path} or {schema_path}")
        return False
    except json.JSONDecodeError:
        print(f"[Failed] File content is not valid JSON format: {json_path}")
        return False


if __name__ == '__main__':
    # Save schema file as svl5_schema.json
    SCHEMA_FILE = "svl5_schema.json"
    
    # Put all tracker-generated json files in a directory, e.g. 'json_v5/'
    data_dir = Path("json_v5") 
    
    all_files = list(data_dir.rglob("*.json"))
    if not all_files:
        print(f"No .json files found in directory '{data_dir}'.")
    
    success_count = 0
    for json_file in all_files:
        if validate_svl_file(json_file, SCHEMA_FILE):
            success_count += 1
            
    print("\n--- Validation Complete ---")
    print(f"Total: {len(all_files)} files, Success: {success_count}, Failed: {len(all_files) - success_count}.")