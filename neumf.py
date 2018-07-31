import numpy as np
from mxnet.gluon import nn
from mxnet import nd
import mxnet as mx
import pdb

# in mxnet ,using mxnet.gluon.nn.Block to initiate the the deep neural network
class NeuMF(nn.HybridBlock): #if using nn.Hybridblock, it will generate static graph
    def __init__(self, nb_users, nb_items,
                 mf_dim, mf_reg,
                 mlp_layer_sizes,
                 mlp_layer_regs,# mlp_layer_regs is a reconfirm
                 ctx):  # Indicate the context is CPU or GPU
        if len(mlp_layer_sizes) != len(mlp_layer_regs):
            raise RuntimeError('u dummy, layer_sizes != layer_regs!')
        if mlp_layer_sizes[0] % 2 != 0:
            raise RuntimeError('u dummy, mlp_layer_sizes[0] % 2 != 0')
        super(NeuMF, self).__init__()
        with self.name_scope():

            nb_mlp_layers = len(mlp_layer_sizes)  # nb is the short of  number

            # TODO: regularization
            # the usage of nn.Embedding: nn.Embedding(num of input neuron, num of output neuron)
            # the usage of nn.Dense: nn.Dense(num of output neuron, activation = 'relu'),
            # while the num of input neurons
            # can be ignored.
            nn.Embedding
            self.mf_user_embed = nn.Embedding(nb_users, mf_dim,
                                          weight_initializer=mx.init.Normal())
            self.mf_item_embed = nn.Embedding(nb_items, mf_dim,
                                          weight_initializer=mx.init.Normal())
                                        # mf_dim means the number of predictive factors,
            self.mlp_user_embed = nn.Embedding(nb_users, mlp_layer_sizes[0] // 2,
                                           weight_initializer=mx.init.Normal())
            # put user and item into the mlp together
            self.mlp_item_embed = nn.Embedding(nb_items, mlp_layer_sizes[0] // 2,
                                           weight_initializer=mx.init.Normal())

            self.mlp = nn.HybridSequential()
            for i in range(1, nb_mlp_layers):
                self.mlp.add(nn.Dense(units=mlp_layer_sizes[i],
                                      in_units=mlp_layer_sizes[i-1],
                                      activation='relu',
                                      weight_initializer=mx.init.Xavier(magnitude=6)))
            self.final = nn.Dense(in_units=mlp_layer_sizes[-1] + mf_dim,
                                  units=1,
                                  weight_initializer=mx.init.Xavier())
                                  # the default magnitude is 3
                                  # the final fully-connected layer

    def forward(self, user, item, sigmoid=False):
        xmfu = self.mf_user_embed(user)  # user vector in matrix factorization
        xmfi = self.mf_item_embed(item)  # item vector in matrix factorization
        xmf = xmfu * xmfi                # use element-wise product to calculate the xmfu and xmfi

        xmlpu = self.mlp_user_embed(user)
        xmlpi = self.mlp_item_embed(item)
        # TODO 3: is there function:cat in mxnet
        # previous one: mlp = torch.cat((xmlpu, xmlpi), dim=1)
        # use nd.concatenate([xmlpu, xmlpi], axis=0) to replace
        xmlp = nd.concat(xmlpu, xmlpi, dim=1)
        # axis=0 means increasing the row, while axis=1 means increasing the column

        xmlp = self.mlp(xmlp)

        x = nd.concat(xmf,xmlp, dim=1)
        x = self.final(x)
        if sigmoid:
            x = nd.sigmoid(x)
        return x

def main():
#jiji
    model = NeuMF(1000, 1000,
                  mf_dim=64, mf_reg=0.,
                  mlp_layer_sizes = [256,128,64],
                  mlp_layer_regs=[0. for i in [256,128,64]],
                  ctx=mx.cpu(0))
    print(model)
    model.initialize()
    a = nd.ones([10])
    b = nd.ones([10])
    model(a,b,True)  # 在调用model时，参数里不要带着参数名，直接写变量就好
    c = model(a,b,True)
    c.as_in_context(mx.cpu(0)).asnumpy()




if __name__ == '__main__':
    main()
