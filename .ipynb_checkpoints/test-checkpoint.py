import torch
from model.FULL_MODEL import ExciPickModel

# change vocab size as needed
model = ExciPickModel(vocab_size=1000)

# dummy batch
api = torch.randn(4, 20)
dose = torch.randn(4)
per_unit = torch.randint(0, 3, (4,))
route = torch.randint(0, 10, (4,))
form = torch.randint(0, 10, (4,))

output = model(api, dose, per_unit, route, form)

print("Output shape:", output.shape)