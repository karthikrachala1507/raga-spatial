import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

checkpoint = torch.load("models/BEATs_iter3_plus_AS2M_finetuned_cpt2.pt", map_location="cpu")

label_dict = checkpoint["label_dict"]
print("label_dict type:", type(label_dict))
print("label_dict length:", len(label_dict))
print()
print("First 20 entries:")
items = list(label_dict.items())
for i in range(min(20, len(items))):
    print(str(items[i][0]) + " -> " + str(items[i][1]))
print()
print("Last 10 entries:")
for i in range(max(0, len(items)-10), len(items)):
    print(str(items[i][0]) + " -> " + str(items[i][1]))