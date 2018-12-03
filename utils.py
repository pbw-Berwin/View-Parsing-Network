import numpy as np, os, time
from six.moves import xrange
import logging
import torch
from torch.autograd import Variable

class Foo(object):
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)
  def __str__(self):
    str_ = ''
    for v in vars(self).keys():
      a = getattr(self, v)
      str__ = str(a)
      str__ = str__.replace('\n', '\n  ')
      str_ += '{:s}: {:s}'.format(v, str__)
      str_ += '\n'
    return str_


def get_flow(t, theta, map_size):
    """
    Rotates the map by theta and translates the rotated map by t.

    Assume that the robot rotates by an angle theta and then moves forward by
    translation t. This function returns the flow field field. For every pixel in
    the new image it tells us which pixel in the original image it came from:
    NewI(x, y) = OldI(flow_x(x,y), flow_y(x,y)).

    Assume there is a point p in the original image. Robot rotates by R and moves
    forward by t.  p1 = Rt*p; p2 = p1 - t; (the world moves in opposite direction.
    So, p2 = Rt*p - t, thus p2 came from R*(p2+t), which is what this function
    calculates.

      t:      ... x 2 (translation for B batches of N motions each).
      theta:  ... x 1 (rotation for B batches of N motions each).

      Output: ... x map_size x map_size x 2
    """
    B = t.view(-1, 2).size()[0]
    tx, ty = torch.unbind(torch.view(t, [-1, 1, 1, 1, 2]), dim=4) # Bx1x1x1
    theta = torch.view(theta, [-1, 1, 1, 1])
    # c = tf.constant((map_size - 1.) / 2., dtype=tf.float32)
    c = Variable(torch.Tensor([(map_size - 1.) / 2.]).double())
    x, y = np.meshgrid(np.arange(map_size[0]), np.arange(map_size[1]))
    x = Variable(x[np.newaxis, :, :, np.newaxis]).view(1, map_size[0], map_size[1], 1)
    y = Variable(y[np.newaxis, :, :, np.newaxis]).view(1, map_size[0], map_size[1], 1)
    # x = tf.constant(x[np.newaxis, :, :, np.newaxis], dtype=tf.float32, name='x',
    #                 shape=[1, map_size, map_size, 1])
    # y = tf.constant(y[np.newaxis, :, :, np.newaxis], dtype=tf.float32, name='y',
    #                 shape=[1, map_size, map_size, 1])

    tx = tx - c.expand(tx.size())
    x = x.expand([B] + x.size()[1:])
    x = x + tx.expand(x.size())
    ty = ty - c.expand(ty.size())
    y = y.expand([B] + y.size()[1:])
    y = y + ty.expand(y.size()) # BxHxWx1
    # x = x - (-tx + c.expand(tx.size())) #1xHxWx1
    # y = y - (-ty + c.expand(ty.size()))

    sin_theta = torch.sin(theta) #Bx1x1x1
    cos_theta = torch.cos(theta)
    xr = x * cos_theta.expand(x.size()) - y * sin_theta.expand(y.size())
    yr = x * sin_theta.expand(x.size()) + y * cos_theta.expand(y.size()) # BxHxWx1
    # xr = cos_theta * x - sin_theta * y
    # yr = sin_theta * x + cos_theta * y

    xr = xr + c.expand(xr.size())
    yr = yr + c.expand(yr.size())

    flow = torch.stack([xr, yr], axis=-1)
    sh = t.size()[:-1] + [map_size[0], map_size[1], 2]
    # sh = tf.unstack(tf.shape(t), axis=0)
    # sh = tf.stack(sh[:-1] + [tf.constant(_, dtype=tf.int32) for _ in [map_size, map_size, 2]])
    flow = torch.view(flow, shape=sh)
    return flow


def dense_resample(im, flow_im, output_valid_mask=False):
    """ Resample reward at particular locations.
    Args:
      im:      ...xHxW matrix to sample from.
      flow_im: ...xHxWx2 matrix, samples the image using absolute offsets as given
               by the flow_im.
    """
    valid_mask = None

    x, y = torch.unbind(flow_im, axis=-1)
    x = x.view(-1)
    y = y.view(-1)

    # constants
    # shape = tf.unstack(tf.shape(im))
    # channels = shape[-1]
    shape = im.size()
    width = shape[-1]
    height = shape[-2]
    num_batch = 1
    for dim in shape[:-2]:
        num_batch *= dim
    zero = Variable(torch.Tensor([0]).double())
    # num_batch = tf.cast(tf.reduce_prod(tf.stack(shape[:-3])), 'int32')
    # zero = tf.constant(0, dtype=tf.int32)

    # Round up and down.
    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

    x0 = x0.clamp(0, width - 1)
    x1 = x1.clamp(0, width - 1)
    y0 = y0.clamp(0, height - 1)
    y1 = y1.clamp(0, height - 1)
    dim2 = width
    dim1 = width * height

    # Create base index
    base = torch.range(num_batch) * dim1
    base = base.view(-1, 1)
    # base = tf.reshape(tf.range(num_batch) * dim1, shape=[-1, 1])
    base = base.expand(base.size()[0], height * width).view(-1) # batch_size * H * W
    # base = tf.reshape(tf.tile(base, [1, height * width]), shape=[-1])

    base_y0 = base + y0.expand(base.size()) * dim2
    base_y1 = base + y1.expand(base.size()) * dim2
    idx_a = base_y0 + x0.expand(base_y0.size())
    idx_b = base_y1 + x0.expand(base_y1.size())
    idx_c = base_y0 + x1.expand(base_y0.size())
    idx_d = base_y1 + x1.expand(base_y1.size())

    # use indices to lookup pixels in the flat image and restore channels dim
    # sh = tf.stack([tf.constant(-1, dtype=tf.int32), channels])
    im_flat = torch.view(im, [-1])
    # im_flat = tf.cast(tf.reshape(im, sh), dtype=tf.float32)
    pixel_a = torch.gather(im_flat, idx_a)
    pixel_b = torch.gather(im_flat, idx_b)
    pixel_c = torch.gather(im_flat, idx_c)
    pixel_d = torch.gather(im_flat, idx_d)

    # and finally calculate interpolated values
    # x1_f = tf.to_float(x1)
    # y1_f = tf.to_float(y1)
    x1_f = x1.float()
    y1_f = y1.float()

    wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
    wb = torch.unsqueeze(((x1_f - x) * (1.0 - (y1_f - y))), 1)
    wc = torch.unsqueeze(((1.0 - (x1_f - x)) * (y1_f - y)), 1)
    wd = torch.unsqueeze(((1.0 - (x1_f - x)) * (1.0 - (y1_f - y))), 1)

    output = wa * pixel_a.unsqueeze(1) + wb * pixel_b.unsqueeze(1) + wc * pixel_c.unsqueeze(1) + wd * pixel_d.unsqueeze(1)
    # output = tf.reshape(output, shape=tf.shape(im))
    output = output.view(im.size())
    return output, valid_mask
