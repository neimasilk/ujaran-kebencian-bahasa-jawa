#!/usr/bin/env python3

import json
import os

def check_threshold_results():
    print("=== CHECKING THRESHOLD TUNING RESULTS ===")
    
    if os.path.exists('threshold_tuning_results.json'):
        try:
            with open('threshold_tuning_results.json', 'r') as f:
                data = json.load(f)
            
            print("\nKeys in threshold_tuning_results.json:")
            for key in data.keys():
                print(f"  - {key}")
            
            print("\nContent:")
            for key, value in data.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for subkey, subvalue in value.items():
                        print(f"    {subkey}: {subvalue}")
                else:
                    print(f"  {key}: {value}")
                    
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        print("File threshold_tuning_results.json not found")

if __name__ == "__main__":
    check_threshold_results()