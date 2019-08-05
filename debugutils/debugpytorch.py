from inferno.extensions.layers.reshape import Concatenate
from inferno.extensions.containers.graph import Graph, Identity


class VerboseConcatenate(Concatenate):

    def forward(self, x, y):
        print("merging....", x.shape, y.shape)
        return super().forward(x, y)


class VerboseIdentity(nn.Module):

    def __init__(self, name):
        self.name = name
        super().__init__()

    def forward(self, x):
        print(self.name, x.shape)
        return x
