import numpy
import chainer.datasets
from PIL import Image


def _resize_image(img, size, dtype):
    # convert grayscale into RGB
    if img.shape[0] == 1:
        imgarray = numpy.asarray([img[0]] * 3, dtype=numpy.uint8)
    else:
        imgarray = numpy.asarray(img, dtype=numpy.uint8)

    imgarray = imgarray.transpose(1, 2, 0)

    pilimg = Image.fromarray(imgarray)
    pilimg = pilimg.resize(size, Image.BICUBIC)

    return numpy.asarray(pilimg, dtype=dtype).transpose(2, 0, 1) / 255


class ImagenetFetchDataset(chainer.datasets.LabeledImageDataset):
    """
    Image dataset which resizes image to fit the model automatically.
    This dataset object read image files each time as needed,
    and does not store whole dataset in memory at once,
    same as chainer.ImageDataset.
    """

    def __init__(self, filelist, size=227, root='.',
                 dtype=numpy.float32, label_dtype=numpy.int32):

        if isinstance(size, int):
            self.insize = (size, size)
        else:
            self.insize = tuple(size)

        super().__init__(filelist, root=root,
                         dtype=dtype, label_dtype=label_dtype)

    def __getitem__(self, index):
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return [self.__getitem__(i) for i in
                    range(current, stop, step)]
        else:
            img, label = super().__getitem__(index)
            return _resize_image(img, self.insize, self._dtype), label


class ImageNetPrefetchDataset(chainer.datasets.TupleDataset):
    """
    Image dataset which resizes image to fit the model automatically.
    This dataset object stores whole data in memory.
    This would not suit for massive image dataset,
    but suit for small size of dataset in terms of the performance.
    """

    def __init__(self, filelist, size=227, root='.',
                 dtype=numpy.float32, label_dtype=numpy.int32):
        fetch_dataset = ImagenetFetchDataset(filelist, size, root, dtype,
                                             label_dtype)

        n_data = len(fetch_dataset)
        whole = fetch_dataset[0:n_data]

        data = [x[0] for x in whole]
        label = [x[1] for x in whole]

        super().__init__(data, label)
