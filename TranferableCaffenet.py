import chainer
import chainer.functions as F
import chainer.links as L


class TransferableCaffenet(chainer.Chain):
    """
    CaffeNet implementation which has named fully-connected layers.
    """

    insize = 227

    def __init__(self, n_out, fc_name=None):
        if fc_name is None:
            self.fc_name = ''
        else:
            self.fc_name = '_' + fc_name

        args = {
            'conv1': L.Convolution2D(3,  96, 11, stride=4),
            'conv2': L.Convolution2D(96, 256,  5, pad=2),
            'conv3': L.Convolution2D(256, 384,  3, pad=1),
            'conv4': L.Convolution2D(384, 384,  3, pad=1),
            'conv5': L.Convolution2D(384, 256,  3, pad=1),
            'fc6' + self.fc_name: L.Linear(9216, 4096),
            'fc7' + self.fc_name: L.Linear(4096, 4096),
            'fc8' + self.fc_name: L.Linear(4096, 1000)
        }

        super(Caffenet, self).__init__(
            **args
        )

        self.layers = args.keys()

        self.train = True

    def fc(self, n):
        name = 'fc' + str(n) + self.fc_name
        return getattr(self, name)

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x):
        self.clear()
        h = F.local_response_normalization(F.max_pooling_2d(
            F.relu(self.conv1(x)), 3, stride=2))
        h = F.local_response_normalization(F.max_pooling_2d(
            F.relu(self.conv2(h)), 3, stride=2))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc(6)(h)))
        h = F.dropout(F.relu(self.fc(7)(h)))
        h = self.fc(8)(h)

        return h
