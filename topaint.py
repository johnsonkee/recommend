import matplotlib.pyplot as plt
import numpy as np
from mxnet import autograd,nd
import mxnet as mx


num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
