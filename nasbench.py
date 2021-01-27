from nats_bench import create
import os
os.environ["TORCH_HOME"] = '/notebooks/storage/.torch'
# Create the API instance for the topology search space in NATS
api = create(None, 'tss', fast_mode=True, verbose=True)

info = api.get_more_info(1234, 'cifar10')

# Query the flops, params, latency. info is a dict.
info = api.get_cost_info(12, 'cifar10')

# Simulate the training of the 1224-th candidate:
validation_accuracy, latency, time_cost, current_total_time_cost = api.simulate_train_eval(1224, dataset='cifar10', hp='12')