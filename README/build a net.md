# 搭建网络
下面使用`gluon`模块来搭建一个简单的网络
## 导入mxnet函数包
```
# import dependencies
from __future__ import print_function
import numpy as np
import mxnet as mx
import mxnet.ndarray as F
import mxnet.gluon as gluon
from mxnet.gluon import nn
from mxnet import autograd    # 自动求导模块
```
`gluon`里有`nn`模块和`rnn`模块，构建多层感知器以及深度神经网络可以直接使用`nn`模块，若是想构建循环神经网络得使用`rnn`模块。
## 定义网络
下面介绍两种定义网络的方式，第一种方式构建简便，但网络结构单一；第二种方式构建过程稍微复杂，但是网络结构更加灵活。
```
# 方式一

```

```
# 方式二
# gluon.Block是最基本的网络
class Net(gluon.Block):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            # layers created in name_scope will inherit name space
            # from parent layer.
            self.conv1 = nn.Conv2D(6, kernel_size=5)
            self.pool1 = nn.MaxPool2D(pool_size=(2,2))
            self.conv2 = nn.Conv2D(16, kernel_size=5)
            self.pool2 = nn.MaxPool2D(pool_size=(2,2))
            self.fc1 = nn.Dense(120)
            self.fc2 = nn.Dense(84)
            self.fc3 = nn.Dense(10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # 0 means copy over size from corresponding dimension.
        # -1 means infer size from the rest of dimensions.
        x = x.reshape((0, -1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
