from nats_bench import create
import os
from tqdm import tqdm
os.environ["TORCH_HOME"] = '/notebooks/storage/.torch'
os.environ["TORCH_HOME"] = r'C:\Users\kawga\.torch'

# Create the API instance for the topology search space in NATS
api = create(None, 'tss', fast_mode=True, verbose=False)

per_epoch = []
for idx in tqdm(range(15000), desc = "Counting avg per epoch trianing time"):
    if idx % 1000 == 0:
        api = create(None, 'tss', fast_mode=True, verbose=False)
    info = api.get_more_info(idx, 'cifar10-valid')
    per_epoch.append(info['train-per-time'])
avg = sum(per_epoch)/len(per_epoch) # The avg is 9.14s per epoch

# Query the flops, params, latency. info is a dict.
info = api.get_cost_info(1223, 'ImageNet16-120')

# Simulate the training of the 1224-th candidate:
validation_accuracy, latency, time_cost, current_total_time_cost = api.simulate_train_eval(8758, dataset='cifar10', hp='12', iepoch=None)