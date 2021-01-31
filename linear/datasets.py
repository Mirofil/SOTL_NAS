import pandas as pd
import torch

def get_datasets(name, path=None):
    if name == "songs":
        if path is None:
            path = r"C:\Users\kawga\Documents\Oxford\thesis\data\YearPredictionMSD.txt"

        data = pd.read_csv(path)
        y = data[data.columns[0]].values.tolist()
        x = data[data.columns[1:]].values.tolist()

        y = torch.tensor(y, dtype=torch.float32)
        x = torch.tensor(x, dtype=torch.float32)

    return x, y