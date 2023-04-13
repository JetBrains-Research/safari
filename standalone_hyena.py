"""
Simplified standalone version of Hyena: https://arxiv.org/abs/2302.10866, designed for quick experimentation.
A complete version is available under `src.models.sequence.hyena`.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def fftconv(u, k, D):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
    
    if len(u.shape) > 3: k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)


@torch.jit.script 
def mul_sum(q, y):
    return (q * y).sum(dim=1)

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
            

class Sin(nn.Module):
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim)) if train_freq else w * torch.ones(1, dim)

    def forward(self, x):
        return torch.sin(self.freq * x)
    
    
class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float=1e-5, **kwargs): 
        """Complex exponential positional embeddings for Hyena filters."""  
        super().__init__()
        
        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None] # 1, L, 1
        
        if emb_dim > 1:
            bands = (emb_dim - 1) // 2            
        # To compute the right embeddings we use the "proper" linspace 
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len # 1, L, 1 
        
        f = torch.linspace(1e-4, bands - 1, bands)[None, None] 
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register("z", z, lr=lr_pos_emb) 
        self.register("t", t, lr=0.0)
        
    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]
    

class ExponentialModulation(OptimModule):
    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulation_lr=0.0,
        modulate: bool=True,
        shift: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register("deltas", deltas, lr=modulation_lr)
        
    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs()) 
            x = x * (decay + self.shift)
        return x                  


class HyenaFilter(OptimModule):
    def __init__(
            self, 
            d_model,
            emb_dim=3, # dim of input to MLP, augments with positional encoding
            order=16, # width of the implicit MLP 
            fused_fft_conv=False,
            seq_len=1024, 
            lr=1e-3, 
            lr_pos_emb=1e-5,
            dropout=0.0, 
            w=1, # frequency of periodic activations 
            wd=0, # weight decay of kernel parameters 
            use_bias=True,
            num_inner_mlps=2,
            normalized=False,
            use_modulation=True,
            **kwargs
        ):
        """
        Implicit long filter with modulation.
        
        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP
        """
        super().__init__()
        self.d_model = d_model
        self.use_bias = use_bias
        self.fused_fft_conv = fused_fft_conv
        if use_bias:
            self.bias = nn.Parameter(torch.randn(self.d_model))
        else:
            self.bias = torch.zeros(self.d_model)
        self.dropout = nn.Dropout(dropout)
        
        act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        assert emb_dim % 2 != 0 and emb_dim >= 3, "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.seq_len = seq_len
  
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)
        
        self.implicit_filter = nn.Sequential(
            nn.Linear(emb_dim, order),
            act,
        )
        for i in range(num_inner_mlps):
            self.implicit_filter.append(nn.Linear(order, order))
            self.implicit_filter.append(act)

        self.implicit_filter.append(nn.Linear(order, d_model, bias=False))
            
        self.use_modulation = use_modulation
        self.modulation = ExponentialModulation(d_model, **kwargs)
        
        self.normalized = normalized
        for c in self.implicit_filter.children():
            for name, v in c.state_dict().items():        
                optim = {"weight_decay": wd, "lr": lr}
                setattr(getattr(c, name), "_optim", optim)

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        if self.use_modulation:
            h = self.modulation(t, h)
        return h

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None: k = self.filter(L)
        
        # Ensure compatibility with filters that return a tuple 
        k = k[0] if type(k) is tuple else k 

        y = fftconv(x, k, bias)
        return y
    
    
class HyenaOperator(nn.Module):
    def __init__(
            self,
            d_model,
            l_max,
            d_context=None,
            l_context_max=None,
            use_context=False,
            order=2, 
            filter_order=64,
            dropout=0.0,  
            filter_dropout=0.0, 
            **filter_args,
        ):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf
        
        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length. Defaults to None
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
        """
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        inner_width = d_model * (order + 1)
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(d_model, inner_width)
        self.out_proj = nn.Linear(d_model, d_model)

        self.short_filter = nn.Conv1d(
            inner_width, 
            inner_width, 
            3,
            padding=2,
            groups=inner_width
        )
        self.filter_fn = HyenaFilter(
            d_model * (order - 1), 
            order=filter_order, 
            seq_len=l_max,
            channels=1, 
            dropout=filter_dropout, 
            **filter_args
        )


        self.use_context = use_context
        if use_context:
            context_width = d_model * 2
            self.d_context = d_context
            self.context_proj = nn.Linear(d_context, context_width)
            # self.context_to_model = nn.Linear(l_max, d_model)
            self.l_context_max = l_context_max


            self.context_short_filter = nn.Conv1d(
                context_width,
                context_width,
                3,
                padding=2,
                groups=context_width
            )
            self.context_filter_fn = HyenaFilter(
                d_model * 2,
                order=filter_order,
                seq_len=l_max,
                channels=1,
                dropout=filter_dropout,
                use_modulation=False,
                w=4
                #**filter_args
            )



    def forward(self, u, in_context=None, *args, **kwargs):
        l = u.size(-2)
        l_filter = min(l, self.l_max)
        u = self.in_proj(u)
        u = rearrange(u, 'b l d -> b d l')
        
        uc = self.short_filter(u)[...,:l_filter] 
        *x, v = uc.split(self.d_model, dim=1)
        
        k = self.filter_fn.filter(l_filter)[0] # 1x(oxd) -> (oxd)
        k = rearrange(k, 'l (o d) -> o d l', o=self.order - 1)
        bias = rearrange(self.filter_fn.bias, '(o d) -> o d', o=self.order - 1)

        # in_context -> Projection -> split -> filtering ->
        if in_context is not None and self.use_context:
            l_context = in_context.size(-2)
            # l_context = min(l_context, self.l_context_max)
            context = self.context_proj(in_context)
            context = rearrange(context, 'b l d -> b d l')
            if l_filter >= l_context:
                context = nn.functional.pad(context, (0, l_filter-l_context), mode='reflect') # TODO: create different strategies for the context padding
            else:
                print('Warning: context length is greater than sequence length\n')

            context_c = self.context_short_filter(context)[..., :l_filter]
            c0, c1 = context_c.split(self.d_model, dim=1)

            # DONE: TODO: instantiate filter with no modulation
            k_context = self.context_filter_fn.filter(l_filter)[0]
            k_context = rearrange(k_context, 'l (o d) -> o d l', o=2)
            bias_context = rearrange(self.context_filter_fn.bias, '(o d) -> o d', o=2)

            # DONE: TODO: append c0, c1 to x, k_context to k, bias_context to bias
            x.append(c0)
            x.append(c1)
            k = torch.cat((k, k_context), dim=0)
            bias = torch.cat((bias, bias_context), dim=0)
            # x = [c0, c1] + x
            # k = torch.cat((k_context, k), dim=0)
            # bias = torch.cat((bias_context, bias), dim=0)

        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o])

        y = rearrange(v * x[0], 'b d l -> b l d')

        y = self.out_proj(y)
        return y


class MyModel(nn.Module):
    def __init__(self,
                 d_model,
                 l_max,
                 order,
                 filter_order,
                 use_context,
                 d_context,
                 **filter_args
                 ):
        super(MyModel, self).__init__()
        self.hyena = HyenaOperator(
            d_model=d_model,
            l_max=l_max,
            order=order,
            filter_order=filter_order,
            use_context=use_context,
            d_context=d_context,
            **filter_args
        )
        self.fc = nn.Linear(d_model, d_context)

    def forward(self, x, ctxt):
        x = self.hyena(x, in_context=ctxt)
        x = self.fc(x)
        return x





if __name__ == "__main__":
    context_dim = 20
    l_max = 10000
    d_model = 20
    dataset_len = 2000
    use_context = True
    num_epochs = 10
    batch_size = 100
    # if not use_context:
    #     l_max *= 2

    import random

    def label_transform(label, length=l_max-10, context_dim=context_dim):

        label = label.item()
        res = []
        for _ in range(length//3):
            wis = random.randint(1, 5)
            res += [label]
            for s in range(wis):
                res += [random.randint(0, context_dim-1)]
            if len(res) > length - length//15:
                res += [random.randint(0, context_dim-1) for _ in range(length-len(res))]
                break
        return res

    def plot_kernel(model):
        import matplotlib.pyplot as plt

        k = model.hyena.filter_fn.filter(l_max)[0]  # 1x(oxd) -> (oxd)

        k = rearrange(k, 'l (o d) -> o d l', o=model.hyena.order-1)
        bias = rearrange(model.hyena.filter_fn.bias, '(o d) -> o d', o=model.hyena.order -1)
        print(k.size(), bias.size())

        _, idx = k.detach().std(dim=-1).max(dim=-1)
        idx = idx[0].item()
        print(idx)
        for i in range(model.hyena.order -1):
            plt.plot(range(l_max), k[i, idx].detach())# + bias[0, idx].detach())
        #plt.plot(range(l_max), k[1, idx].detach())# + bias[1, idx].detach())
        if model.hyena.use_context:
            k_context = model.hyena.context_filter_fn.filter(l_max)[0]
            k_context = rearrange(k_context, 'l (o d) -> o d l', o=2)
            bias_context = rearrange(model.hyena.context_filter_fn.bias, '(o d) -> o d', o=2)
            plt.plot(range(l_max), k_context[0, idx].detach())# + bias_context[0, idx].detach())
            plt.plot(range(l_max), k_context[1, idx].detach())# + bias_context[1, idx].detach())
        plt.show()


    model = MyModel(
        d_model=d_model,
        l_max=l_max if use_context else l_max*2,
        order=3,
        filter_order=64,
        use_context=use_context,
        d_context=context_dim,
        w=30,
        emb_dim=33
    )
    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    # Train the model

    sequences = [(2.0 + 5.0 * torch.randn((1, l_max, d_model), requires_grad=True)).softmax(dim=-1).detach() for _
                 in range(dataset_len)]
    # labels = [torch.randint(context_dim, (1)) for _ in range(dataset_len)]
    labels = torch.randint(context_dim, (dataset_len,)).detach()
    context = [torch.eye(context_dim)[label_transform(label)].unsqueeze(0).detach().requires_grad_() for label in
               labels]
    if not use_context:
        sequences = [torch.cat((seq, cont), dim=1) for cont, seq in zip(context, sequences)]

    for epoch in range(num_epochs):
        if epoch%1==0:
            plot_kernel(model)
        running_loss = 0.0
        correct = 0
        total = 0



        for i in range(0, len(sequences), batch_size):
            # Get the current batch
            inputs = sequences[i:i + batch_size]
            x = torch.cat(inputs, dim=0)
            ctxt = context[i:i + batch_size]
            ctxt = torch.cat(ctxt, dim=0)
            lbls = labels[i:i + batch_size]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(x, ctxt)[:, -1, :]
            #outputs = model(ctxt, x)[:, -1, :]
            # print(outputs.size())
            # print(outputs[:, -1, :].size())
            loss = criterion(outputs, lbls)
            running_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += lbls.size(0)
            correct += (predicted == lbls).sum().item()

            model.eval()
            with torch.no_grad():

                t_sequences = [(5.0 - 4.0 * torch.randn((1, l_max, d_model), requires_grad=True)).softmax(dim=-1) for _
                               in
                               range(100)]
                # labels = [torch.randint(context_dim, (1)) for _ in range(dataset_len)]
                t_labels = torch.randint(context_dim, (100,))

                t_context = [torch.eye(context_dim)[label_transform(label)].unsqueeze(0).requires_grad_() for label in
                             t_labels]

                if not use_context:
                    t_sequences = [torch.cat((seq, cont), dim=1) for cont, seq in zip(t_context, t_sequences)]

                y_pred = torch.max(
                    model(
                        torch.cat(t_sequences, dim=0),
                          torch.cat(t_context, dim=0)
                          )[:, -1, :],
                    dim=1
                )[1]
                t_correct = (y_pred == t_labels).sum().item()
                print(f'after {i} sequences Acc: {t_correct/100 :.4f}')

                #loss_val = criterion(y_pred_val.squeeze(), y_val.float())

            model.train()

        # Compute and print the epoch loss and accuracy
        epoch_loss = running_loss / (len(sequences) / batch_size)
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}")

    plot_kernel(model)




