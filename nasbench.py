from nats_bench import create
import os
os.environ["TORCH_HOME"] = r'C:\Users\kawga\Documents\Oxford\thesis\playground'

# Create the API instance for the topology search space in NATS
api = create(None, 'tss', fast_mode=False, verbose=True)