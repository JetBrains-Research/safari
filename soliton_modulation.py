import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def squared_sech(x, amp, width, phase):
    """
    Computes the square of the hyperbolic secant of a tensor x.
    """
    # width = torch.tensor(width, dtype=torch.float32)
    # amp = torch.tensor(amp, dtype=torch.float32)
    # phase = torch.tensor(phase, dtype=torch.float32)
    cosh = torch.cosh(torch.reciprocal(width)*(x - phase))
    sech = torch.reciprocal(cosh)
    #return torch.exp(amp)*torch.pow(sech, 2)
    return amp*torch.pow(sech, 2)

class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None: optim["lr"] = lr
            if wd is not None: optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)

class SolitonModulation(OptimModule):
    def __init__(
            self,
            d_model,
            fast_decay_pct=0.3,
            slow_decay_pct=1.5,
            target=1e-2,
            modulation_lr=0.0,
            modulate: bool = True,
            shift: float = 0.0,
            amp=1.,
            width=1.,
            phase=0.,
            **kwargs
    ):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        self.amp = amp
        self.width = width
        self.phase = phase
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        #print(min_decay, max_decay)
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        #print(deltas)
        self.register("deltas", deltas, lr=modulation_lr)

    def forward(self, t, x):
        if self.modulate:
            decay = squared_sech(t, self.amp, self.width, self.phase)
            #print(decay)
            x = x * (decay+self.shift)
        return x


seq_len = 10000
# modulation_1 = SolitonModulation(1, amp=0, width=0.1, phase=1.)
# modulation_2 = SolitonModulation(1, amp=-1, width=0.2, phase=0.)
# modulation_3 = SolitonModulation(1, amp=0.5, width=0.01, phase=0.5)
t = torch.linspace(0, 1, seq_len)[None, :, None]
x = torch.zeros((seq_len,))[None, :, None]
ones = 1. + torch.zeros((seq_len,))[None, :, None]
for _ in range(10):
    modulation = SolitonModulation(1, amp=1.0+torch.randn((1,)), width=0.1*torch.randn((1,)), phase=0.5 + 0.5* torch.randn((1,)))
    x += modulation(t, ones)
    #plt.plot(t[0, :, 0], x[0, :, 0])


#plt.plot(t[0,:,0], x[0,:,0])
# x = modulation_1(t, ones)
# plt.plot(t[0,:,0], x[0,:,0])
# x += modulation_2(t, ones)
# plt.plot(t[0,:,0], x[0,:,0])
# x += modulation_3(t, ones)
# plt.plot(t[0,:,0], x[0,:,0])
#print(x)
#import matplotlib.pyplot as plt
plt.plot(t[0,:,0], x[0,:,0])
plt.show()