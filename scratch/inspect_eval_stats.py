import pickle

pkl_path = ""

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("Keys:", data.keys())
for key in data.keys():
    print(f"\n--- {key} ---")
    sub_data = data[key]
    print("Subkeys:", sub_data.keys())
    for subkey in sub_data.keys():
        val = sub_data[subkey]
        if isinstance(val, list):
            print(f"  {subkey}: list of length {len(val)}, first 3 elements: {val[:3]}")
        elif isinstance(val, dict):
            print(f"  {subkey}: dict, keys: {val.keys()}")
        else:
            print(f"  {subkey}: {type(val)}")