# implementation of Zeiler's max-unpooling, which differ from
# chainer.functions.max_unpooling
# This version of max-unpooling pads underiverred region with 0,
# in other words, ignore pixel values except the maximum element
# in each receptive fields.

# [Reference]
# Zeiler, M. D., & Fergus, R. (2014).
# Visualizing and understanding convolutional networks.
# Lecture Notes in Computer Science, 818–833.


# TODO: Do not use cv2
import numpy
cv2 = None


def receptive_fields(shape, ksize, stride, DEBUG=False):
    height, width = shape[1:3]

    # reference point
    # y, x : index of original image
    # j, i : index of pooled image
    y, x = 0, 0
    j, i = 0, 0

    while j < height:
        while i < width:

            yield y, x, j, i
            x += stride
            i += 1

        y += stride
        j += 1

        x = 0
        i = 0


def max_unpooling_2d(orig, h, ksize, stride, pad, outsize, DEBUG=False):
    n_batch, n_channels, height, width = h.shape

    ret = numpy.zeros((n_batch, n_channels, outsize[0], outsize[1]), dtype=numpy.float32)

    i_batch = 0

    for oimg_, himg, vret in zip(orig, h, ret):
        oimg = cv2.copyMakeBorder(oimg_, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

        rf = receptive_fields(himg.shape, ksize, stride, DEBUG)

        for y, x, j, i in rf:
            val = himg[:, j, i]
            lookup = oimg[:, y:y + ksize, x:x + ksize]
            maxinds = [tuple([k]) + numpy.unravel_index(p.argmax(), p.shape) for k, p in enumerate(lookup)]

            for k_channel, index in enumerate(maxinds):
                ret[i_batch, k_channel, y + index[1], x + index[2]] = val[k_channel]

        i_batch += 1

    return ret
