import numpy as np
from mxnet.gluon import nn
from mxnet import nd

# in mxnet ,using mxnet.gluon.nn.Block to initiate the the deep neural network
class NeuMF(nn.Block): #if using nn.Hybridblock, it will generate static graph
    def __init__(self, nb_users, nb_items,
                 mf_dim, mf_reg,
                 mlp_layer_sizes, mlp_layer_regs):  # mlp_layer_regs is a reconfirm
        if len(mlp_layer_sizes) != len(mlp_layer_regs):
            raise RuntimeError('u dummy, layer_sizes != layer_regs!')
        if mlp_layer_sizes[0] % 2 != 0:
            raise RuntimeError('u dummy, mlp_layer_sizes[0] % 2 != 0')
        super(NeuMF, self).__init__()
        nb_mlp_layers = len(mlp_layer_sizes)  # nb is the short of  number

        # TODO: regularization?
        # TODO 1: problem：Using embedding or Dense
        # the usage of nn.Embedding: nn.Embedding(num of input neuron, num of output neuron)
        # the usage of nn.Dense: nn.Dense(num of output neuron, activation = 'relu'), while the num of input neurons
        # can be ignored.
        self.mf_user_embed = nn.Embedding(nb_users, mf_dim)
        self.mf_item_embed = nn.Embedding(nb_items, mf_dim) # mf_dim means the number of predictive factors,
        self.mlp_user_embed = nn.Embedding(nb_users, mlp_layer_sizes[0] // 2)  # put user and item into the mlp together
        self.mlp_item_embed = nn.Embedding(nb_items, mlp_layer_sizes[0] // 2)

        self.mlp = nn.Sequential()
        for i in range(1, nb_mlp_layers):
            self.mlp.add([nn.Dense(mlp_layer_sizes[i])])  # noqa: E501 # previous one is in
                                                                                                   # latter one is out
        self.final = nn.Embedding(mlp_layer_sizes[-1] + mf_dim, 1) # the final fully-connected layer
        self.mf_user_embed.weight.data = nd.random.normal(0., 0.01, self.mf_user_embed.weight.shape)
        self.mf_item_embed.weight.data = nd.random.normal(0., 0.01, self.mf_item_embed.weight.shape)
        self.mlp_user_embed.weight.data = nd.random.normal(0., 0.01, self.mlp_user_embed.weight.shape)
        self.mlp_item_embed.weight.data = nd.random.normal(0., 0.01, self.mlp_item_embed.weight.shape)
        # golorot_uniform is one of a initialization, always used in fc layers, convolution layers, preventing its
        # saturation
        def golorot_uniform(layer):
            # TODO 2: what is layer.in_feature
            fan_out, fan_in = layer.weight.shape  #
            limit = np.sqrt(6. / (fan_in + fan_out))
            layer.weight.data = nd.uniform(-limit, limit)

        def lecunn_uniform(layer):
            fan_out, fan_in = layer.weight.shape  # noqa: F841, E501
            limit = np.sqrt(3. / fan_in)
            layer.weight.data = nd.uniform(-limit, limit)

        for layer in self.mlp:
            if type(layer) != nn.Dense:
                continue
            golorot_uniform(layer)
        lecunn_uniform(self.final)

    def forward(self, user, item, sigmoid=False):
        xmfu = self.mf_user_embed(user)  # user vector in matrix factorization
        xmfi = self.mf_item_embed(item)  # item vector in matrix factorization
        xmf = xmfu * xmfi                # use element-wise product to calculate the xmfu and xmfi

        xmlpu = self.mlp_user_embed(user)
        xmlpi = self.mlp_item_embed(item)
        # TODO 3: is there function:cat in mxnet
        # previous one: mlp = torch.cat((xmlpu, xmlpi), dim=1)
        # use nd.concatenate([xmlpu, xmlpi], axis=0) to replace
        xmlp = nd.concatenate([xmlpu, xmlpi], axis=1)
        for i, layer in enumerate(self.mlp):
            xmlp = layer(xmlp)
            xmlp = nd.relu(xmlp)

        x = nd.concatenate([xmf, xmlp], axis=0)
        x = self.final(x)
        if sigmoid:
            x = nd.sigmoid(x)


def main():

    model = NeuMF(138493, 26744,
                  mf_dim=64, mf_reg=0.,
                  mlp_layer_sizes = [256,128,64],
                  mlp_layer_regs=[0. for i in [256,128,64]] )
    print(model)

if __name__ == '__main__':
    main()
