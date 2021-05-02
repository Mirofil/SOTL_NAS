from utils import data_generator
from layers import LinearMaxDeg

data=data_generator(10, max_order=5, max_order_y=5, noise_var=0, featurize_type="polynomial")
assert sum(sum(data[0][0]) == data[1][0]) == 1

data2=data_generator(10, max_order=5, max_order_y=3, noise_var=0, featurize_type="polynomial")
assert round(sum(data[0][0])) == round(sum(data2[0][0]))
assert round(sum(data2[0][0][0:3])) == round(sum(data2[1][0]))


a = LinearMaxDeg(10,2)
a.squished_tanh(torch.tensor([5], dtype=torch.float32),plot=True)