import sys
sys.path.append('..')
import chainer
import chainer.links as L

from TransferableCaffenet import TransferableCaffenet


dst = sys.argv[1]
model = L.Classifier(TransferableCaffenet(1000, 'dummy'))
model.to_cpu()
chainer.serializers.save_npz(dst, model)
