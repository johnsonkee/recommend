import mxnet as mx
import torch as tc
import time


time1 = time.time()
a = mx.nd.ones([1000,1])
b = mx.nd.ones([1000,1])
c = a+b
time2 = time.time()
print(time2 - time1)



time1 = time.time()
a = tc.tensor([1000,1])
b = tc.tensor([1000,1])
c = a+b
time2 = time.time()
print(time2 - time1)


