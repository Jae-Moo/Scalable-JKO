import torch
import torch.nn as nn

import collections

import torch
import torch.nn.functional as F

from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast
from functools import partial
import queue
import collections
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P

__all__ = ['FutureResult', 'SlavePipe', 'SyncMaster']


class FutureResult(object):
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, 'Previous result has\'t been fetched.'
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()

            res = self._result
            self._result = None
            return res


_MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])
_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier', 'queue', 'result'])


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


class SyncMaster(object):
    """An abstract `SyncMaster` object.

    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """

        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def __getstate__(self):
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        self.__init__(state['master_callback'])

    def register_slave(self, identifier):
        """
        Register an slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.

        """
        if self._activated:
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).

        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.

        Returns: the message to be sent back to the master device.

        """
        self._activated = True

        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())

        results = self._master_callback(intermediates)
        assert results[0][0] == 0, 'The first result should belongs to the master.'

        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)

        for i in range(self.nr_slaves):
            assert self._queue.get() is True

        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)


__all__ = ['SynchronizedBatchNorm1d', 'SynchronizedBatchNorm2d', 'SynchronizedBatchNorm3d']


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """add new dementions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum', 'sum_size'])
_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])
# _MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'ssum', 'sum_size'])

class _SynchronizedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)

        self._sync_master = SyncMaster(self._data_parallel_master)

        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input, gain=None, bias=None):
        # If it is not parallel computation or is in evaluation mode, use PyTorch's implementation.
        if not (self._is_parallel and self.training):
            out = F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)
            if gain is not None:
              out = out + gain
            if bias is not None:
              out = out + bias
            return out

        # Resize the input to (B, C, -1).
        input_shape = input.size()
        # print(input_shape)
        input = input.view(input.size(0), input.size(1), -1)

        # Compute the sum and square-sum.
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)
        # Reduce-and-broadcast the statistics.
        # print('it begins')
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))
        # if self._parallel_id == 0:
            # # print('here')
            # sum, ssum, num = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        # else:
            # # print('there')
            # sum, ssum, num = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))
        
        # print('how2')
        # num = sum_size
        # print('Sum: %f, ssum: %f, sumsize: %f, insum: %f' %(float(sum.sum().cpu()), float(ssum.sum().cpu()), float(sum_size), float(input_sum.sum().cpu()))) 
        # Fix the graph
        # sum = (sum.detach() - input_sum.detach()) + input_sum
        # ssum = (ssum.detach() - input_ssum.detach()) + input_ssum
        
        # mean = sum / num
        # var = ssum / num - mean ** 2
        # # var = (ssum - mean * sum) / num
        # inv_std = torch.rsqrt(var + self.eps)
        
        # Compute the output.
        if gain is not None:
          # print('gaining')
          # scale = _unsqueeze_ft(inv_std) * gain.squeeze(-1)
          # shift = _unsqueeze_ft(mean) * scale - bias.squeeze(-1)
          # output = input * scale - shift
          output = (input - _unsqueeze_ft(mean)) * (_unsqueeze_ft(inv_std) * gain.squeeze(-1)) + bias.squeeze(-1)
        elif self.affine:
            # MJY:: Fuse the multiplication for speed.
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)        
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)

        # Reshape it.
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id

        # parallel_id == 0 means master device.
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""

        # Always using same "device order" makes the ReduceAdd operation faster.
        # Thanks to:: Tete Xiao (http://tetexiao.com/)
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())

        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # flatten
        target_gpus = [i[1].sum.get_device() for i in intermediates]

        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)

        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)
        # print('a')
        # print(type(sum_), type(ssum), type(sum_size), sum_.shape, ssum.shape, sum_size)
        # broadcasted = Broadcast.apply(target_gpus, sum_, ssum, torch.tensor(sum_size).float().to(sum_.device))
        # print('b')
        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i*2:i*2+2])))
            # outputs.append((rec[0], _MasterMessage(*broadcasted[i*3:i*3+3])))

        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data
        return mean, torch.rsqrt(bias_var + self.eps)
        # return mean, bias_var.clamp(self.eps) ** -0.5


class SynchronizedBatchNorm1d(_SynchronizedBatchNorm):
    r"""Applies Synchronized Batch Normalization over a 2d or 3d input that is seen as a
    mini-batch.

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm1d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal BatchNorm

    Args:
        num_features: num_features from an expected input of size
            `batch_size x num_features [x width]`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm1d, self)._check_input_dim(input)


class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    r"""Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm2d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)


class SynchronizedBatchNorm3d(_SynchronizedBatchNorm):
    r"""Applies Batch Normalization over a 5d input that is seen as a mini-batch
    of 4d inputs

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm3d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric BatchNorm
    or Spatio-temporal BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x depth x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45, 10))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm3d, self)._check_input_dim(input)
def weights_init_xavier(m):
    classname = m.__class__.__name__
    if (classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1) and (classname.find('Cond') == -1) and (classname.find('Spectral') == -1):
        try:
            # Normal conv layer
            nn.init.xavier_uniform_(m.weight)
        except:
            # Conv layer with spectral norm
            nn.init.xavier_uniform_(m.weight_u)
            nn.init.xavier_uniform_(m.weight_v)
            nn.init.xavier_uniform_(m.weight_bar)

    elif classname.find('BatchNorm') != -1 and classname.find('cond') == -1 and classname.find('Cond') == -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def form_onehot(labels, num_classes, device='cuda:0'):
    batch_size = labels.size(0)
    y = torch.FloatTensor(batch_size, num_classes).fill_(0).to(device)

    for i in range(batch_size):
        y[i][labels[i]] = 1
    return y

''' Layers
    This file contains various layers for the BigGAN models.
'''


# Projection of x onto y
def proj(x, y):
  return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
  for y in ys:
    x = x - proj(x, y)
  return x


# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
  # Lists holding singular vectors and values
  us, vs, svs = [], [], []
  for i, u in enumerate(u_):
    # Run one step of the power iteration
    with torch.no_grad():
      v = torch.matmul(u, W)
      # Run Gram-Schmidt to subtract components of all other singular vectors
      v = F.normalize(gram_schmidt(v, vs), eps=eps)
      # Add to the list
      vs += [v]
      # Update the other singular vector
      u = torch.matmul(v, W.t())
      # Run Gram-Schmidt to subtract components of all other singular vectors
      u = F.normalize(gram_schmidt(u, us), eps=eps)
      # Add to the list
      us += [u]
      if update:
        u_[i][:] = u
    # Compute this singular value and add it to the list
    svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
  return svs, us, vs


# Convenience passthrough function
class identity(nn.Module):
  def forward(self, input):
    return input
 

# Spectral normalization base class 
class SN(object):
  def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
    # Number of power iterations per step
    self.num_itrs = num_itrs
    # Number of singular values
    self.num_svs = num_svs
    # Transposed?
    self.transpose = transpose
    # Epsilon value for avoiding divide-by-0
    self.eps = eps
    # Register a singular vector for each sv
    for i in range(self.num_svs):
      self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
      self.register_buffer('sv%d' % i, torch.ones(1))
  
  # Singular vectors (u side)
  @property
  def u(self):
    return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

  # Singular values; 
  # note that these buffers are just for logging and are not used in training. 
  @property
  def sv(self):
   return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]
   
  # Compute the spectrally-normalized weight
  def W_(self):
    W_mat = self.weight.view(self.weight.size(0), -1)
    if self.transpose:
      W_mat = W_mat.t()
    # Apply num_itrs power iterations
    for _ in range(self.num_itrs):
      svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps) 
    # Update the svs
    if self.training:
      with torch.no_grad(): # Make sure to do this in a no_grad() context or you'll get memory leaks!
        for i, sv in enumerate(svs):
          self.sv[i][:] = sv     
    return self.weight / svs[0]


# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1, bias=True, 
             num_svs=1, num_itrs=1, eps=1e-12):
    nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, 
                     padding, dilation, groups, bias)
    SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)    
  def forward(self, x):
    return F.conv2d(x, self.W_(), self.bias, self.stride, 
                    self.padding, self.dilation, self.groups)


# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
  def __init__(self, in_features, out_features, bias=True,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Linear.__init__(self, in_features, out_features, bias)
    SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
  def forward(self, x):
    return F.linear(x, self.W_(), self.bias)


# Embedding layer with spectral norm
# We use num_embeddings as the dim instead of embedding_dim here
# for convenience sake
class SNEmbedding(nn.Embedding, SN):
  def __init__(self, num_embeddings, embedding_dim, padding_idx=None, 
               max_norm=None, norm_type=2, scale_grad_by_freq=False,
               sparse=False, _weight=None,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Embedding.__init__(self, num_embeddings, embedding_dim, padding_idx,
                          max_norm, norm_type, scale_grad_by_freq, 
                          sparse, _weight)
    SN.__init__(self, num_svs, num_itrs, num_embeddings, eps=eps)
  def forward(self, x):
    return F.embedding(x, self.W_())


# A non-local block as used in SA-GAN
# Note that the implementation as described in the paper is largely incorrect;
# refer to the released code for the actual implementation.
class Attention(nn.Module):
  def __init__(self, ch, which_conv=SNConv2d, name='attention'):
    super(Attention, self).__init__()
    # Channel multiplier
    self.ch = ch
    self.which_conv = which_conv
    self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
    self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
    # Learnable gain parameter
    self.gamma = P(torch.tensor(0.), requires_grad=True)
  def forward(self, x, y=None):
    # Apply convs
    theta = self.theta(x)
    phi = F.max_pool2d(self.phi(x), [2,2])
    g = F.max_pool2d(self.g(x), [2,2])    
    # Perform reshapes
    theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
    phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
    g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
    # Attention map times g path
    o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
    return self.gamma * o + x


# Fused batchnorm op
def fused_bn(x, mean, var, gain=None, bias=None, eps=1e-5):
  # Apply scale and shift--if gain and bias are provided, fuse them here
  # Prepare scale
  scale = torch.rsqrt(var + eps)
  # If a gain is provided, use it
  if gain is not None:
    scale = scale * gain
  # Prepare shift
  shift = mean * scale
  # If bias is provided, use it
  if bias is not None:
    shift = shift - bias
  return x * scale - shift
  #return ((x - mean) / ((var + eps) ** 0.5)) * gain + bias # The unfused way.


# Manual BN
# Calculate means and variances using mean-of-squares minus mean-squared
def manual_bn(x, gain=None, bias=None, return_mean_var=False, eps=1e-5):
  # Cast x to float32 if necessary
  float_x = x.float()
  # Calculate expected value of x (m) and expected value of x**2 (m2)  
  # Mean of x
  m = torch.mean(float_x, [0, 2, 3], keepdim=True)
  # Mean of x squared
  m2 = torch.mean(float_x ** 2, [0, 2, 3], keepdim=True)
  # Calculate variance as mean of squared minus mean squared.
  var = (m2 - m **2)
  # Cast back to float 16 if necessary
  var = var.type(x.type())
  m = m.type(x.type())
  # Return mean and variance for updating stored mean/var if requested  
  if return_mean_var:
    return fused_bn(x, m, var, gain, bias, eps), m.squeeze(), var.squeeze()
  else:
    return fused_bn(x, m, var, gain, bias, eps)


# My batchnorm, supports standing stats    
class myBN(nn.Module):
  def __init__(self, num_channels, eps=1e-5, momentum=0.1):
    super(myBN, self).__init__()
    # momentum for updating running stats
    self.momentum = momentum
    # epsilon to avoid dividing by 0
    self.eps = eps
    # Momentum
    self.momentum = momentum
    # Register buffers
    self.register_buffer('stored_mean', torch.zeros(num_channels))
    self.register_buffer('stored_var',  torch.ones(num_channels))
    self.register_buffer('accumulation_counter', torch.zeros(1))
    # Accumulate running means and vars
    self.accumulate_standing = False
    
  # reset standing stats
  def reset_stats(self):
    self.stored_mean[:] = 0
    self.stored_var[:] = 0
    self.accumulation_counter[:] = 0
    
  def forward(self, x, gain, bias):
    if self.training:
      out, mean, var = manual_bn(x, gain, bias, return_mean_var=True, eps=self.eps)
      # If accumulating standing stats, increment them
      if self.accumulate_standing:
        self.stored_mean[:] = self.stored_mean + mean.data
        self.stored_var[:] = self.stored_var + var.data
        self.accumulation_counter += 1.0
      # If not accumulating standing stats, take running averages
      else:
        self.stored_mean[:] = self.stored_mean * (1 - self.momentum) + mean * self.momentum
        self.stored_var[:] = self.stored_var * (1 - self.momentum) + var * self.momentum
      return out
    # If not in training mode, use the stored statistics
    else:         
      mean = self.stored_mean.view(1, -1, 1, 1)
      var = self.stored_var.view(1, -1, 1, 1)
      # If using standing stats, divide them by the accumulation counter   
      if self.accumulate_standing:
        mean = mean / self.accumulation_counter
        var = var / self.accumulation_counter
      return fused_bn(x, mean, var, gain, bias, self.eps)


# Simple function to handle groupnorm norm stylization                      
def groupnorm(x, norm_style):
  # If number of channels specified in norm_style:
  if 'ch' in norm_style:
    ch = int(norm_style.split('_')[-1])
    groups = max(int(x.shape[1]) // ch, 1)
  # If number of groups specified in norm style
  elif 'grp' in norm_style:
    groups = int(norm_style.split('_')[-1])
  # If neither, default to groups = 16
  else:
    groups = 16
  return F.group_norm(x, groups)


# Class-conditional bn
# output size is the number of channels, input size is for the linear layers
# Andy's Note: this class feels messy but I'm not really sure how to clean it up
# Suggestions welcome! (By which I mean, refactor this and make a pull request
# if you want to make this more readable/usable). 
class ccbn(nn.Module):
  def __init__(self, output_size, input_size, which_linear, eps=1e-5, momentum=0.1,
               cross_replica=False, mybn=False, norm_style='bn',):
    super(ccbn, self).__init__()
    self.output_size, self.input_size = output_size, input_size
    # Prepare gain and bias layers
    self.gain = which_linear(input_size, output_size)
    self.bias = which_linear(input_size, output_size)
    # epsilon to avoid dividing by 0
    self.eps = eps
    # Momentum
    self.momentum = momentum
    # Use cross-replica batchnorm?
    self.cross_replica = cross_replica
    # Use my batchnorm?
    self.mybn = mybn
    # Norm style?
    self.norm_style = norm_style
    
    if self.cross_replica:
      self.bn = SynchronizedBatchNorm2d(output_size, eps=self.eps, momentum=self.momentum, affine=False)
    elif self.mybn:
      self.bn = myBN(output_size, self.eps, self.momentum)
    elif self.norm_style in ['bn', 'in']:
      self.register_buffer('stored_mean', torch.zeros(output_size))
      self.register_buffer('stored_var',  torch.ones(output_size)) 
    
    
  def forward(self, x, y):
    # Calculate class-conditional gains and biases
    gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
    bias = self.bias(y).view(y.size(0), -1, 1, 1)
    # If using my batchnorm
    if self.mybn or self.cross_replica:
      return self.bn(x, gain=gain, bias=bias)
    # else:
    else:
      if self.norm_style == 'bn':
        out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
                          self.training, 0.1, self.eps)
      elif self.norm_style == 'in':
        out = F.instance_norm(x, self.stored_mean, self.stored_var, None, None,
                          self.training, 0.1, self.eps)
      elif self.norm_style == 'gn':
        out = groupnorm(x, self.normstyle)
      elif self.norm_style == 'nonorm':
        out = x
      return out * gain + bias
  def extra_repr(self):
    s = 'out: {output_size}, in: {input_size},'
    s +=' cross_replica={cross_replica}'
    return s.format(**self.__dict__)


# Normal, non-class-conditional BN
## litu
# check with large momentum...
# default 0.1
#
class bn(nn.Module):
  def __init__(self, output_size,  eps=1e-5, momentum=0.1,
                cross_replica=False, mybn=False):
    super(bn, self).__init__()
    self.output_size= output_size
    # Prepare gain and bias layers
    self.gain = P(torch.ones(output_size), requires_grad=True)
    self.bias = P(torch.zeros(output_size), requires_grad=True)
    # epsilon to avoid dividing by 0
    self.eps = eps
    # Momentum
    self.momentum = momentum
    # Use cross-replica batchnorm?
    self.cross_replica = cross_replica
    # Use my batchnorm?
    self.mybn = mybn
    
    if self.cross_replica:
      self.bn = SynchronizedBatchNorm2d(output_size, eps=self.eps, momentum=self.momentum, affine=False)    
    elif mybn:
      self.bn = myBN(output_size, self.eps, self.momentum)
     # Register buffers if neither of the above
    else:     
      self.register_buffer('stored_mean', torch.zeros(output_size))
      self.register_buffer('stored_var',  torch.ones(output_size))
    
  def forward(self, x, y=None):
    if self.cross_replica or self.mybn:
      gain = self.gain.view(1,-1,1,1)
      bias = self.bias.view(1,-1,1,1)
      return self.bn(x, gain=gain, bias=bias)
    else:
      return F.batch_norm(x, self.stored_mean, self.stored_var, self.gain,
                          self.bias, self.training, self.momentum, self.eps)

                          
# Generator blocks
# Note that this class assumes the kernel size and padding (and any other
# settings) have been selected in the main generator module and passed in
# through the which_conv arg. Similar rules apply with which_bn (the input
# size [which is actually the number of channels of the conditional info] must 
# be preselected)
class GBlock(nn.Module):
  def __init__(self, in_channels, out_channels,
               which_conv=nn.Conv2d, which_bn=bn, activation=None, 
               upsample=None):
    super(GBlock, self).__init__()
    
    self.in_channels, self.out_channels = in_channels, out_channels
    self.which_conv, self.which_bn = which_conv, which_bn
    self.activation = activation
    self.upsample = upsample
    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.out_channels)
    self.conv2 = self.which_conv(self.out_channels, self.out_channels)
    self.learnable_sc = in_channels != out_channels or upsample
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels, 
                                     kernel_size=1, padding=0)
    # Batchnorm layers
    self.bn1 = self.which_bn(in_channels)
    self.bn2 = self.which_bn(out_channels)
    # upsample layers
    self.upsample = upsample

  def forward(self, x, y):
    h = self.activation(self.bn1(x, y))
    if self.upsample:
      h = self.upsample(h)
      x = self.upsample(x)
    h = self.conv1(h)
    h = self.activation(self.bn2(h, y))
    h = self.conv2(h)
    if self.learnable_sc:       
      x = self.conv_sc(x)
    return h + x
    
    
# Residual block for the discriminator
class DBlock(nn.Module):
  def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True,
               preactivation=False, activation=None, downsample=None,):
    super(DBlock, self).__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
    self.hidden_channels = self.out_channels if wide else self.in_channels
    self.which_conv = which_conv
    self.preactivation = preactivation
    self.activation = activation
    self.downsample = downsample
        
    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
    self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
    self.learnable_sc = True if (in_channels != out_channels) or downsample else False
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels, 
                                     kernel_size=1, padding=0)
  def shortcut(self, x):
    if self.preactivation:
      if self.learnable_sc:
        x = self.conv_sc(x)
      if self.downsample:
        x = self.downsample(x)
    else:
      if self.downsample:
        x = self.downsample(x)
      if self.learnable_sc:
        x = self.conv_sc(x)
    return x
    
  def forward(self, x):
    if self.preactivation:
      # h = self.activation(x) # NOT TODAY SATAN
      # Andy's note: This line *must* be an out-of-place ReLU or it 
      #              will negatively affect the shortcut connection.
      h = F.relu(x)
    else:
      h = x    
    h = self.conv1(h)
    h = self.conv2(self.activation(h))
    if self.downsample:
      h = self.downsample(h)     
        
    return h + self.shortcut(x)
    
# dogball

## RobustOT
###############################################################################
# Unconditional layers

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super().__init__()
        self.conv_layer = torch.nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                 padding=padding, dilation=dilation, groups=groups,
                 bias=bias, padding_mode=padding_mode))

    def forward(self, input):
        out = self.conv_layer(input)
        return out


class SpectralLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear_layer = torch.nn.utils.spectral_norm(nn.Linear(in_features, out_features, bias=bias))

    def forward(self, input):
        out = self.linear_layer(input)
        return out


class LeakyReLU2(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=inplace)

    def forward(self, input):
        out = self.lrelu(input)
        return out


conv_layers = {
    'conv': nn.Conv2d,
    'spectral_conv': SpectralConv2d
}

linear_layers = {
    'linear': nn.Linear,
    'spectral_linear': SpectralLinear
}

activation_layers = {
    'relu': nn.ReLU,
    'lrelu': LeakyReLU2
}

norm_layers = {
    'BN': nn.BatchNorm2d,
    'identity': nn.Identity
}

###############################################################################
# Layers with conditional support


class CategoricalConditionalBatchNorm(torch.nn.Module):
    # as in the chainer SN-GAN implementation, we keep per-cat weight and bias
    def __init__(self, num_features, num_cats, eps=2e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.num_cats = num_cats
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(num_cats, num_features))
            self.bias = torch.nn.Parameter(torch.Tensor(num_cats, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.zero_()

    def forward(self, inp):
        input = inp[0]
        cats = inp[1]
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        out = torch.nn.functional.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        if self.affine:
            shape = [input.size(0), self.num_features] + (input.dim() - 2) * [1]
            weight = self.weight.index_select(0, cats).view(shape)
            bias = self.bias.index_select(0, cats).view(shape)
            out = out * weight + bias
        return (out, cats)

    def extra_repr(self):
        return '{num_features}, num_cats={num_cats}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class CondConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                 padding=padding, dilation=dilation, groups=groups,
                 bias=bias, padding_mode=padding_mode)

    def forward(self, input):
        img = input[0]
        label = input[1]
        out = self.conv_layer(img)
        return (out, label)


class CondConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros'):
        super().__init__()
        self.conv_transpose_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                 padding=padding, output_padding=output_padding, groups=groups, bias=bias,
                 dilation=dilation, padding_mode=padding_mode)

    def forward(self, input):
        img = input[0]
        label = input[1]
        out = self.conv_transpose_layer(img)
        return (out, label)


class UnconditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes=10, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                 track_running_stats=track_running_stats)

    def forward(self, input):
        img = input[0]
        label = input[1]
        out = self.batch_norm(img)
        return (out, label)


class CondReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, input):
        img = input[0]
        label = input[1]
        out = self.relu(img)
        return (out, label)


class CondLeakyReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=inplace)

    def forward(self, input):
        img = input[0]
        label = input[1]
        out = self.lrelu(img)
        return (out, label)


class CondTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()

    def forward(self, input):
        img = input[0]
        label = input[1]
        out = self.tanh(img)
        return (out, label)


class CondUpsample(nn.Module):
    def __init__(self, scale_factor=1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor)

    def forward(self, input):
        img = input[0]
        label = input[1]
        out = self.upsample(img)
        return (out, label)


class CondSpectralLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear_layer = torch.nn.utils.spectral_norm(nn.Linear(in_features, out_features, bias=bias))

    def forward(self, input):
        img, label = input
        out = self.linear_layer(img)
        return out, label


class CondLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear_layer = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        img, label = input
        out = self.linear_layer(img)
        return out, label


cond_norm_layers = {
    'BN': UnconditionalBatchNorm2d,
    'conditionalBN': CategoricalConditionalBatchNorm
}

cond_conv_layers = {
    'conv': CondConv2d,
    'convT': CondConvTranspose2d,
    'spectral_conv': None,
    'spectral_convT': None
}

cond_activation_layers = {
    'relu': CondReLU,
    'lrelu': CondLeakyReLU,
    'tanh': CondTanh
}

cond_linear_layers = {
    'linear': CondLinear,
    'spectral_linear': CondSpectralLinear
}


class BaseDiscriminator(nn.Module):
    def __init__(self, config):
        super(BaseDiscriminator, self).__init__()

        self.config = config
        linear_layer = linear_layers[config.D_linear]

        if self.config.conditioning == 'projection':
            self.projection = linear_layer(config.num_classes, config.projection_dim)
        elif self.config.conditioning == 'concat':
            self.projection = linear_layer(config.num_classes + config.projection_dim, 1)
        elif self.config.conditioning == 'acgan':
            self.classifier = linear_layer(config.projection_dim, config.num_classes)
            self.discriminator = linear_layer(config.projection_dim, 1)

    def project(self, input, label):
        label = form_onehot(label, self.config.num_classes, device=input.device)
        if self.config.conditioning == 'projection':
            projection = self.projection(label)
            dot_product = projection * input
            out = torch.sum(dot_product, dim=1)
        elif self.config.conditioning == 'concat':
            inp_cat = torch.cat((input, label), dim=1)
            out = self.projection(inp_cat)
        elif self.config.conditioning == 'acgan':
            disc_logits = self.discriminator(input)
            class_logits = self.classifier(input)
            out = (disc_logits, class_logits)
        return out

class ResBlockGenerator(nn.Module):
    def __init__(self, config, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        conv_layer = cond_conv_layers[config.G_conv]
        norm_layer = cond_norm_layers[config.G_normalization]
        activation_layer = cond_activation_layers[config.G_activation]

        self.model = nn.Sequential(
            norm_layer(in_channels, config.num_classes),
            activation_layer(True),
            CondUpsample(scale_factor=2),
            conv_layer(in_channels, out_channels, 3, 1, padding=1),
            norm_layer(out_channels, config.num_classes),
            activation_layer(True),
            conv_layer(out_channels, out_channels, 3, 1, padding=1)
        )
        weights_init_xavier(self.model)
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = CondUpsample(scale_factor=2)

    def forward(self, x):
        lab = x[1]
        model_out, _ = self.model(x)
        bypass_out, _ = self.model(x)
        return (model_out + bypass_out, lab)


class ResBlockDiscriminator(nn.Module):
    def __init__(self, config, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        conv_layer = conv_layers[config.D_conv]
        activation_layer = activation_layers[config.D_activation]

        if stride == 1:
            self.model = nn.Sequential(
                activation_layer(True),
                conv_layer(in_channels, out_channels, 3, 1, padding=1),
                activation_layer(True),
                conv_layer(out_channels, out_channels, 3, 1, padding=1)
            )
        else:
            self.model = nn.Sequential(
                activation_layer(True),
                conv_layer(in_channels, out_channels, 3, 1, padding=1),
                activation_layer(True),
                conv_layer(out_channels, out_channels, 3, 1, padding=1),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
        weights_init_xavier(self.model)

        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Sequential(
                conv_layer(in_channels,out_channels, 1, 1, padding=0),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
            weights_init_xavier(self.bypass)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):
    def __init__(self, config, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        conv_layer = conv_layers[config.D_conv]
        activation_layer = activation_layers[config.D_activation]

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            conv_layer(in_channels, out_channels, 3, 1, padding=1),
            activation_layer(True),
            conv_layer(out_channels, out_channels, 3, 1, padding=1),
            nn.AvgPool2d(2)
        )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            conv_layer(in_channels, out_channels, 1, 1, padding=0),
        )
        weights_init_xavier(self.model)
        weights_init_xavier(self.bypass)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.z_dim = config.nz
        ngf = self.ngf = config.ngf
        config.conditonal = False
        inp_dim = self.z_dim

        conv_layer = cond_conv_layers[config.G_conv]
        norm_layer = cond_norm_layers['BN']
        activation_layer = cond_activation_layers[config.G_activation]
        lin_layer = cond_linear_layers[config.G_linear]

        self.init_size = int(config.imageSize / (2 ** 3))
        self.dense = lin_layer(inp_dim, self.init_size * self.init_size * ngf)

        self.network = nn.Sequential(
            ResBlockGenerator(config, ngf, ngf, stride=2),
            ResBlockGenerator(config, ngf, ngf, stride=2),
            ResBlockGenerator(config, ngf, ngf, stride=2),
            norm_layer(ngf, config.num_classes),
            activation_layer(True),
            conv_layer(ngf, config.nc, 3, stride=1, padding=1),
            CondTanh()
        )
        weights_init_xavier(self.network)

    def forward(self, input_noise, z, label=None):
        input_noise = input_noise.view(input_noise.shape[0],-1)
        layer1_out, _ = self.dense((input_noise, label))
        layer1_out = layer1_out.view(-1, self.ngf, self.init_size, self.init_size)
        output = self.network((layer1_out, label))
        out, _ = output
        return out


class Discriminator(BaseDiscriminator):
    def __init__(self, config):
        super(Discriminator, self).__init__(config)
        self.config = config
        ndf = config.ngf
        linear_layer = linear_layers[config.D_linear]
        out_dim = 1

        self.feat_net = nn.Sequential(
            FirstResBlockDiscriminator(config, config.nc, ndf, stride=2),
            ResBlockDiscriminator(config, ndf, ndf, stride=2),
            ResBlockDiscriminator(config, ndf, ndf),
            ResBlockDiscriminator(config, ndf, ndf),
            nn.ReLU(),
        )

        self.final_layer = nn.Sequential(
            linear_layer(ndf, out_dim)
        )
        self.final_layer.apply(weights_init_xavier)

    def forward(self, input, label=None):
        feat = self.feat_net(input)
        feat = torch.sum(feat, (2, 3))
        feat = feat.view(feat.size(0), -1)
        disc_logits = self.final_layer(feat)
        return disc_logits



class Generator_64x64(nn.Module):
    def __init__(self, config):
        super(Generator_64x64, self).__init__()
        self.z_dim = config.nz
        self.ngpu = config.ngpu
        ngf = self.ngf = config.ngf
        self.config = config

        
        inp_dim = self.z_dim

        conv_layer = cond_conv_layers[config.G_conv]
        norm_layer = cond_norm_layers['BN']
        activation_layer = cond_activation_layers[config.G_activation]
        lin_layer = cond_linear_layers[config.G_linear]

        self.init_size = int(config.imageSize / (2 ** 4))
        self.dense = lin_layer(inp_dim, self.init_size * self.init_size * ngf)

        self.network = nn.Sequential(
            ResBlockGenerator(config, ngf, ngf, stride=2),
            ResBlockGenerator(config, ngf, ngf, stride=2),
            ResBlockGenerator(config, ngf, ngf, stride=2),
            ResBlockGenerator(config, ngf, ngf, stride=2),            
            norm_layer(ngf, 1),
            activation_layer(True),
            conv_layer(ngf, config.nc, 3, stride=1, padding=1),
            CondTanh()
        )
        weights_init_xavier(self.network)

    def forward(self, input_noise, label=None):
        input_noise = input_noise.view(input_noise.shape[0],-1)
        layer1_out, _ = self.dense((input_noise, label))
        layer1_out = layer1_out.view(-1, self.ngf, self.init_size, self.init_size)
        output = self.network((layer1_out, label))
        out, _ = output
        return out


class Discriminator_64x64(BaseDiscriminator):
    def __init__(self, config):
        super(Discriminator_64x64, self).__init__(config)

        self.config = config
        ndf = config.ndf
        
        linear_layer = linear_layers[config.D_linear]

        out_dim = config.projection_dim
        if not config.conditional:
            out_dim = 1

        self.feat_net = nn.Sequential(
            FirstResBlockDiscriminator(config, config.nc, ndf, stride=2),
            ResBlockDiscriminator(config, ndf, ndf, stride=2),
            ResBlockDiscriminator(config, ndf, ndf, stride=2),            
            ResBlockDiscriminator(config, ndf, ndf),
            ResBlockDiscriminator(config, ndf, ndf),
            nn.ReLU(),
        )

        self.final_layer = nn.Sequential(
            linear_layer(ndf, out_dim)
        )
        self.final_layer.apply(weights_init_xavier)

    def forward(self, input, label=None):
        feat = self.feat_net(input)
        feat = torch.sum(feat, (2, 3))
        feat = feat.view(feat.size(0), -1)
        disc_logits = self.final_layer(feat)

        if self.config.conditional:
            disc_logits = self.project(disc_logits, label)

        return disc_logits



class Generator_128x128(nn.Module):
    def __init__(self, config):
        super(Generator_128x128, self).__init__()
        self.z_dim = config.nz
        self.ngpu = config.ngpu
        ngf = self.ngf = config.ngf
        self.config = config

        if config.conditional and (config.conditioning == 'concat' or config.conditioning == 'acgan'):
            inp_dim = self.z_dim + config.num_classes
        else:
            inp_dim = self.z_dim

        conv_layer = cond_conv_layers[config.G_conv]
        norm_layer = cond_norm_layers['BN']
        activation_layer = cond_activation_layers[config.G_activation]
        lin_layer = cond_linear_layers[config.G_linear]

        self.init_size = int(config.imageSize / (2 ** 4))
        self.dense = lin_layer(inp_dim, self.init_size * self.init_size * ngf)

        self.network = nn.Sequential(
            ResBlockGenerator(config, ngf, ngf, stride=2),
            ResBlockGenerator(config, ngf, ngf, stride=2),
            ResBlockGenerator(config, ngf, ngf, stride=2),
            ResBlockGenerator(config, ngf, ngf, stride=2),
            norm_layer(ngf, config.num_classes),
            activation_layer(True),
            conv_layer(ngf, config.nc, 3, stride=1, padding=1),
            CondTanh()
        )
        weights_init_xavier(self.network)

    def forward(self, input_noise, label=None):
        input_noise = input_noise.view(input_noise.shape[0],-1)

        if self.config.conditional and (self.config.conditioning == 'concat' or self.config.conditioning == 'acgan'):
            assert label is not None
            label_onehot = form_onehot(label, self.config.num_classes, device=input_noise.device)
            input_noise = torch.cat((input_noise, label_onehot), dim=1)
        layer1_out, _ = self.dense((input_noise, label))
        layer1_out = layer1_out.view(-1, self.ngf, self.init_size, self.init_size)
        output = self.network((layer1_out, label))
        out, _ = output
        return out


class Discriminator_128x128(BaseDiscriminator):
    def __init__(self, config):
        super(Discriminator_128x128, self).__init__(config)

        self.config = config
        ndf = config.ndf
        
        linear_layer = linear_layers[config.D_linear]

        out_dim = config.projection_dim
        if not config.conditional:
            out_dim = 1

        self.feat_net = nn.Sequential(
            FirstResBlockDiscriminator(config, config.nc, ndf, stride=2),
            ResBlockDiscriminator(config, ndf, ndf, stride=2),
            ResBlockDiscriminator(config, ndf, ndf, stride=2),
            ResBlockDiscriminator(config, ndf, ndf),
            ResBlockDiscriminator(config, ndf, ndf),
            nn.ReLU(),
        )

        self.final_layer = nn.Sequential(
            linear_layer(ndf, out_dim)
        )
        self.final_layer.apply(weights_init_xavier)

    def forward(self, input, label=None):
        feat = self.feat_net(input)
        feat = torch.sum(feat, (2, 3))
        feat = feat.view(feat.size(0), -1)
        disc_logits = self.final_layer(feat)

        if self.config.conditional:
            disc_logits = self.project(disc_logits, label)

        return disc_logits



class Generator_256x256(nn.Module):
    def __init__(self, config):
        super(Generator_256x256, self).__init__()
        self.z_dim = config.nz
        self.ngpu = config.ngpu
        ngf = self.ngf = config.ngf
        self.config = config

        if config.conditional and (config.conditioning == 'concat' or config.conditioning == 'acgan'):
            inp_dim = self.z_dim + config.num_classes
        else:
            inp_dim = self.z_dim

        conv_layer = cond_conv_layers[config.G_conv]
        norm_layer = cond_norm_layers['BN']
        activation_layer = cond_activation_layers[config.G_activation]
        lin_layer = cond_linear_layers[config.G_linear]

        self.init_size = int(config.imageSize / (2 ** 5))
        self.dense = lin_layer(inp_dim, self.init_size * self.init_size * ngf)

        self.network = nn.Sequential(
            ResBlockGenerator(config, ngf, ngf, stride=2),
            ResBlockGenerator(config, ngf, ngf, stride=2),
            ResBlockGenerator(config, ngf, ngf, stride=2),
            ResBlockGenerator(config, ngf, ngf, stride=2),
            ResBlockGenerator(config, ngf, ngf, stride=2),
            norm_layer(ngf, config.num_classes),
            activation_layer(True),
            conv_layer(ngf, config.nc, 3, stride=1, padding=1),
            CondTanh()
        )
        weights_init_xavier(self.network)

    def forward(self, input_noise, label=None):
        input_noise = input_noise.view(input_noise.shape[0],-1)

        if self.config.conditional and (self.config.conditioning == 'concat' or self.config.conditioning == 'acgan'):
            assert label is not None
            label_onehot = form_onehot(label, self.config.num_classes, device=input_noise.device)
            input_noise = torch.cat((input_noise, label_onehot), dim=1)
        layer1_out, _ = self.dense((input_noise, label))
        layer1_out = layer1_out.view(-1, self.ngf, self.init_size, self.init_size)
        output = self.network((layer1_out, label))
        out, _ = output
        return out


class Discriminator_256x256(BaseDiscriminator):
    def __init__(self, config):
        super(Discriminator_256x256, self).__init__(config)

        self.config = config
        ndf = config.ndf
        
        linear_layer = linear_layers[config.D_linear]

        out_dim = config.projection_dim
        if not config.conditional:
            out_dim = 1

        self.feat_net = nn.Sequential(
            FirstResBlockDiscriminator(config, config.nc, ndf, stride=2),
            ResBlockDiscriminator(config, ndf, ndf, stride=2),
            ResBlockDiscriminator(config, ndf, ndf, stride=2),
            ResBlockDiscriminator(config, ndf, ndf, stride=2),
            ResBlockDiscriminator(config, ndf, ndf),
            ResBlockDiscriminator(config, ndf, ndf),
            nn.ReLU(),
        )

        self.final_layer = nn.Sequential(
            linear_layer(ndf, out_dim)
        )
        self.final_layer.apply(weights_init_xavier)

    def forward(self, input, label=None):
        feat = self.feat_net(input)
        feat = torch.sum(feat, (2, 3))
        feat = feat.view(feat.size(0), -1)
        disc_logits = self.final_layer(feat)

        if self.config.conditional:
            disc_logits = self.project(disc_logits, label)

        return disc_logits

def spectral_norm(layer, n_iters=1):
    return torch.nn.utils.spectral_norm(layer, n_power_iterations=n_iters)

class UpsampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, spec_norm=False):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
        if spec_norm:
            self.conv = spectral_norm(self.conv)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, inputs):
        output = inputs
        output = torch.cat([output, output, output, output], dim=1)
        output = self.pixelshuffle(output)
        return self.conv(output)


class InstanceNorm2dPlus(nn.Module):
    def __init__(self, num_features, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.gamma = nn.Parameter(torch.zeros(num_features))
        self.alpha.data.normal_(1, 0.02)
        self.gamma.data.normal_(1, 0.02)
        if bias:
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = self.gamma.view(-1, self.num_features, 1, 1) * h + self.beta.view(-1, self.num_features, 1, 1)
        else:
            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = self.gamma.view(-1, self.num_features, 1, 1) * h
        return out


def conv1x1(in_planes, out_planes, stride=1, bias=True, spec_norm=False):
    "1x1 convolution"
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)
    if spec_norm:
        conv = spectral_norm(conv)
    return conv


def conv3x3(in_planes, out_planes, stride=1, bias=True, spec_norm=False):
    "3x3 convolution with padding"
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)
    if spec_norm:
        conv = spectral_norm(conv)

    return conv


def stride_conv3x3(in_planes, out_planes, kernel_size, bias=True, spec_norm=False):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2,
                     padding=kernel_size // 2, bias=bias)
    if spec_norm:
        conv = spectral_norm(conv)
    return conv


def dilated_conv3x3(in_planes, out_planes, dilation, bias=True, spec_norm=False):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=dilation, dilation=dilation, bias=bias)
    if spec_norm:
        conv = spectral_norm(conv)

    return conv

class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, adjust_padding=False, spec_norm=False):
        super().__init__()
        if not adjust_padding:
            conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
            if spec_norm:
                conv = spectral_norm(conv)
            self.conv = conv
        else:
            conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
            if spec_norm:
                conv = spectral_norm(conv)

            self.conv = nn.Sequential(
                nn.ZeroPad2d((1, 0, 1, 0)),
                conv
            )

    def forward(self, inputs):
        output = self.conv(inputs)
        output = sum([output[:, :, ::2, ::2], output[:, :, 1::2, ::2],
                      output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
        return output

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, resample=None, act=nn.ReLU(),
                 normalization=nn.BatchNorm2d, adjust_padding=False, dilation=None, spec_norm=False):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalization = normalization
        if resample == 'down':
            if dilation is not None:
                self.conv1 = dilated_conv3x3(input_dim, input_dim, dilation=dilation, spec_norm=spec_norm)
                self.normalize2 = normalization(input_dim)
                self.conv2 = dilated_conv3x3(input_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation, spec_norm=spec_norm)
            else:
                self.conv1 = conv3x3(input_dim, input_dim, spec_norm=spec_norm)
                self.normalize2 = normalization(input_dim)
                self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding, spec_norm=spec_norm)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding, spec_norm=spec_norm)

        elif resample is None:
            if dilation is not None:
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation, spec_norm=spec_norm)
                self.conv1 = dilated_conv3x3(input_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
                self.normalize2 = normalization(output_dim)
                self.conv2 = dilated_conv3x3(output_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
            else:
                # conv_shortcut = nn.Conv2d ### Something wierd here.
                conv_shortcut = partial(conv1x1, spec_norm=spec_norm)
                self.conv1 = conv3x3(input_dim, output_dim, spec_norm=spec_norm)
                self.normalize2 = normalization(output_dim)
                self.conv2 = conv3x3(output_dim, output_dim, spec_norm=spec_norm)
        else:
            raise Exception('invalid resample value')

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim)


    def forward(self, x):
        output = self.normalize1(x)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output

##########################################################

class ResNet_G(torch.nn.Module):
    def __init__(self, latent_dim=192, out_channels=3, features=256):
        super().__init__()        
        self.begin_conv = nn.Sequential(
            nn.Conv2d(in_channels=latent_dim, out_channels=features, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(features, affine=True,  track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.trans_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=features, out_channels=features*2, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(features*2, affine=True,  track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.up1 = nn.Sequential(
            UpsampleConv(features*2,features*4),
            nn.BatchNorm2d(features*4, affine=True,  track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.up2 = nn.Sequential(
            UpsampleConv(features*4,features*4),
            nn.BatchNorm2d(features*4, affine=True,  track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True)
            )

        self.up3 = nn.Sequential(
            UpsampleConv(features*4,features*2),
            nn.BatchNorm2d(features*2, affine=True,  track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True)
            )

        self.up4 = nn.Sequential(
            UpsampleConv(features*2,features*2),
            nn.BatchNorm2d(features*2, affine=True,  track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True)
            )


        self.end_conv = nn.Conv2d(in_channels=features*2, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            
        self.output = nn.Tanh()

    def forward(self, x, z):
        x = x.view(x.size(0),x.size(1),1,1)
        x = self.begin_conv(x)
        x = self.trans_conv(x)

        up1 = self.up1(x)
        up2 = self.up2(up1)
        up3 = self.up3(up2)
        up4 = self.up4(up3)

        op = self.end_conv(up4) 
        op = self.output(op)

        return op

""" ResNet_D from NCSN """
class ResNet_D(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=128):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.norm = InstanceNorm2dPlus

        self.begin_conv = nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3,stride=1,padding=1)

        self.down1 = nn.Sequential(
            ResidualBlock(features, features*2, resample='down', act=self.act, normalization=self.norm)
            )
        self.down2 = nn.Sequential(
            ResidualBlock(features*2, features*4, resample='down', act=self.act, normalization=self.norm)
            )
        self.down3 = nn.Sequential(
            ResidualBlock(features*4, features*2, resample='down', act=self.act, normalization=self.norm)
            )
        self.down4 = nn.Sequential(
            ResidualBlock(features*2, features, resample='down', act=self.act, normalization=self.norm)
            )

        self.end_conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=4, stride=1, padding=0)

    
    def forward(self, x):
        x = self.begin_conv(x)

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        op = self.end_conv(x)
        return op