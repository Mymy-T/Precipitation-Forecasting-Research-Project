from model import Model
import os
import torch


"""
Perform a quick check to ensure that the magnitudes of the weights actually differ between wd=1e-4 and wd=1e-5; obtained a minor difference of 0.020403132773935795
"""

model_path_1 = os.path.join(os.getcwd(), "Hyperparameter_Testing_Models", "model__hparam_tune__1_sites__lb5__nh32__wd0.0001")
model_path_2 = os.path.join(os.getcwd(), "Hyperparameter_Testing_Models", "model__hparam_tune__1_sites__lb5__nh32__wd1e-05")

model1_state = torch.load(model_path_1)
model2_state = torch.load(model_path_2)

model1 = Model(15, 1, 32)
model2 = Model(15, 1, 32)
model1.load_state_dict(model1_state)
model2.load_state_dict(model2_state)

total_diff = sum(
    (p1 - p2).norm().item()
    for p1, p2 in zip(model1.parameters(), model2.parameters())
)

print(total_diff)