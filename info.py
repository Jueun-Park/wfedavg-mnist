num_model_5_indices = [list(range(10))[i:i+4] for i in range(0, 8, 2)] + [[8, 9, 0, 1]]
num_model_4_indices = [list(range(10))[i:i+4] for i in range(0, 8, 2)]
num_model_5_comments = [str(i)+"-"+str(i+4) for i in range(0, 8, 2)] + ["8-2"]
num_model_4_comments = [str(i)+"-"+str(i+4) for i in range(0, 8, 2)]

model_indices = num_model_5_indices
model_comments = num_model_5_comments

num_models = 5
base_learn_epochs = 5
fed_learn_epochs = 1
grid_size = 8
