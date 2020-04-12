from os import system

program = "wfedavg_save_clients.py"
num_clients = 4
for i in range(num_clients):
    system(f"python {program} --base-index {i}")
