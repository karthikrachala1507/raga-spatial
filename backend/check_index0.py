import torch
checkpoint = torch.load(
    "models/BEATs_iter3_plus_AS2M_finetuned_cpt2.pt",
    map_location="cpu"
)
label_dict = checkpoint["label_dict"]
print("Index 0 MID:", label_dict[0])
print("Index 1 MID:", label_dict[1])
print("Index 2 MID:", label_dict[2])
print("Index 3 MID:", label_dict[3])
print("Index 4 MID:", label_dict[4])
print()
# Print all indices where confidence was high in our test
# The dominant one was index 0 at 87-95%
# Let's find what /m/ ID that actually is
for i in range(20):
    print(str(i) + " -> " + str(label_dict[i]))