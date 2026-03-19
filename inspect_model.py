import torch
from pathlib import Path

model_path = r"c:\Users\HUNG\Downloads\data-20260312T234957Z-1-001\antigravity_system\backend\data\v47_snapshot_ep72.pt"
try:
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    print(f"Type: {type(state_dict)}")
    if isinstance(state_dict, dict):
        print(f"Keys: {list(state_dict.keys())[:10]}")
        # If it's a nested dict, print one level deeper
        for k in ['model_state_dict', 'state_dict', 'model']:
            if k in state_dict:
                print(f"Found nested key '{k}', sub-keys: {list(state_dict[k].keys())[:10]}")
    else:
        print("Not a dictionary.")
except Exception as e:
    print(f"Error: {e}")
