from os import system
from info import num_models

# make it to multi process,,, not a for sentence

if __name__ == "__main__":
    program = "wfedavg_load_and_fed.py"
    for base_index in range(num_models):
        system(f"python {program} --base-index {base_index}")
