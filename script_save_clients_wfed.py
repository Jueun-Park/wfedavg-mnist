from os import system
from info import num_models

program = "wfedavg_save_clients.py"

for i in range(num_models):
    system(f"python {program} --base-index {i}")
