# WFedAvg MNIST

## Execution Sequence
### WFedAvg Weight Grid Search
```bash
python learner_train_base_models.py
python simple_base_agent_test.py
python wfedavg_save_clients.py
python wfedavg_load_and_fed.py
```

### Train and Test GAN Discriminator
```bash
python gan_train_four_gans.py
python gan_load_and_test_gan.py
```
