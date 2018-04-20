from functools import reduce
import chainer
import numpy
from warnings import warn


class EqualizeIterator(chainer.iterators.SerialIterator):
    """
    Dataset iterator which equalize number of class samples with
    replacement sampling.
    This is used for imbalanced dataset to adjust the model averages.

    Args:
        dataset: Dataset to iterate
        batch_size (int): length of minibatch.
            batch_size should be multiple of a number of target labels,
            if not so, adjusted automatically.
            And this must be less than the number of least example label.
        repeat (bool): same as SerialIterator.
            But it is expected that the same number of mini-batch will be given
            in the last iteration.
        shuffle (bool): same as SerialIterator.
    """

    def __init__(self, dataset, batch_size, repeat=True, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self._repeat = repeat
        self._shuffle = shuffle

        label = list()
        for i in range(len(dataset)):
            # store labels into label
            label.append(dataset[i][1])

        self.label = sorted(list(set([int(x) for x in label])))
        self.label_indices = {lab: list() for lab in self.label}

        for i, lab in enumerate(label):
            self.label_indices[int(lab)].append(i)

        # validate batch size
        if batch_size % len(self.label) != 0:
            self.batch_size += len(self.label) - (batch_size % len(self.label))
            warn("batch_size must be multiple of a number of labels. "
                 "{} would be carried up to {}".format(
                     batch_size, self.batch_size))

        # step width of an iteration
        self.step = int(self.batch_size / len(self.label))

        for key, value in self.label_indices.items():
            if self.step > len(value):
                raise ValueError('Label {} does not have enough items for '
                                 'specified batch size. 1-iteration makes '
                                 'duplicated samples from data.'.format(
                                     str(key)))

        self.reset()

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self._previous_epoch_detail = self.epoch_detail

        batch = list()

        current_inds = self.current_position

        for label, inds in self.label_indices.items():
            Ninds = len(inds)
            current_pos = current_inds[label]
            batch_inds = inds[current_pos:current_pos + self.step]

            if current_pos + self.step >= Ninds:
                rest = current_pos + self.step - Ninds
                if self._shuffle:
                    numpy.random.shuffle(inds)
                if rest > 0:
                    batch_inds.extend(inds[:rest])

                self.current_position[label] = rest
                self.completion_flags[label] = True
            else:
                self.current_position[label] = current_pos + self.step

            batch.extend(batch_inds)

        if numpy.asarray(list(self.completion_flags.values())).all():
            self.epoch += 1
            self.is_new_epoch = True
            self.completion_flags = {label: False for label in self.label}
        else:
            self.is_new_epoch = False

        if self._shuffle:
            numpy.random.shuffle(batch)

        batch_data = [self.dataset[index] for index in batch]

        return batch_data

    next = __next__

    def reset(self):
        self.current_position = {label: 0 for label in self.label}
        self.epoch = 0
        self.is_new_epoch = False
        self.completion_flags = {label: False for label in self.label}

        if self._shuffle:
            for value in self.label_indices.values():
                numpy.random.shuffle(value)

    @property
    def epoch_detail(self):
        return self.epoch

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None

        return self._previous_epoch_detail
