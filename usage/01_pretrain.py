import sys
sys.path.append('..')
import numpy
import chainer
import chainer.links as L
from chainer.training import extensions

from TransferableCaffenet import TransferableCaffenet
from equalize_iterator import EqualizeIterator
from imagenet_dataset import ImagenetFetchDataset, ImageNetPrefetchDataset
from copy_model import copy_model


filelist_path = sys.argv[1]
model_param_path = sys.argv[2]
result_dst_path = sys.argv[3]
onmemory = bool(int(sys.argv[4]))

if onmemory:
    ImagenetDataset = ImageNetPrefetchDataset
else:
    ImagenetDataset = ImagenetFetchDataset

chainer.cuda.get_device_from_id(0).use()

dataset = ImagenetDataset(filelist_path)
train, test = chainer.datasets.split_dataset_random(
    dataset, int(len(dataset) * 0.95)
)

train_iter = chainer.iterators.SerialIterator(train, 1024)
test_iter = chainer.iterators.SerialIterator(test, 1024)

dummy_model = L.Classifier(TransferableCaffenet(1000))
# chainer.serializers.load_npz(model_param_path, dummy_model)
with numpy.load(model_param_path) as f:
    d = chainer.serializers.NpzDeserializer(f, strict=False)
    d.load(dummy_model)

model = L.Classifier(TransferableCaffenet(2, fc_name='targ'))
copy_model(dummy_model.predictor, model.predictor)

model.to_gpu()

optimizer = chainer.optimizers.MomentumSGD(0.005)
optimizer.setup(model)

updater = chainer.training.StandardUpdater(train_iter, optimizer, device=0)
trainer = chainer.training.Trainer(updater, (1000, 'epoch'), out=result_dst_path)
trainer.extend(extensions.Evaluator(test_iter, model, device=0))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.ProgressBar())
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy']
))
trainer.extend(chainer.training.extensions.snapshot(), trigger=(50, 'epoch'))
trainer.run()

model.to_cpu()
chainer.serializers.save_hdf5(result_dst_path + '/result.model.h5', model)
