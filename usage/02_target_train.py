import sys
import numpy
import chainer
import chainer.links as L

from TransferableCaffenet import TransferableCaffenet
from equalize_iterator import EqualizeIterator
from imagenet_dataset import ImagenetFetchDataset, ImageNetPrefetchDataset
from copy_model import copy_model


train_filelist_path = sys.argv[1]
test_filelist_path = sys.argv[2]
model_param_path = sys.argv[3]
result_dst_path = sys.argv[4]
onmemory = bool(int(sys.argv[5]))

if onmemory:
    ImagenetDataset = ImageNetPrefetchDataset
else:
    ImagenetDataset = ImagenetFetchDataset

train = ImagenetDataset(train_filelist_path)
test = ImagenetDataset(test_filelist_path)

train_iter = EqualizeIterator(train, 256)
test_iter = chainer.iterators.SerialIterator(test, 256)

dummy_model = L.Classifier(TransferableCaffenet(1000))
chainer.serializers.load_hdf5(model_param_path, dummy_model)
model = L.Classifier(TransferableCaffenet(61))
copy_model(dummy_model, model)

optimizer = chainer.optimizers.MomentumSGD(0.005)
updater = chainer.training.StandardUpdater(train_iter, optimizer)
trainer = chainer.training.Trainer(updater, (1000, 'epoch'), out=result_dst_path)
trainer.extend(chainer.training.extensions.Evaluator(test_iter, model))
trainer.extend(chainer.training.extensions.LogReport())
trainer.extend(chainer.training.extensions.ProgressBar())
trainer.extend(chainer.training.extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy']
))
trainer.extend(chainer.training.extensions.snapshot(), trigger=(100, 'epoch'))

chainer.serializers.save_hdf5(result_dst_path + '/result.model.h5')
