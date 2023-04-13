import math

from re import U
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from einops import rearrange, repeat

try:
    from src.ops.fftconv import fftconv_ref, fftconv_func 
except ImportError:
    fftconv_func = None

try:
    from flash_attn.ops.fused_dense import FusedDense
except ImportError:
    FusedDense = None

import sys
sys.path.append('./')
import src.utils.registry as registry
from src.utils.train import OptimModule
from src.utils.config import instantiate, auto_assign_attrs
from src.models.nn import Activation


# reference convolution with residual connection
def fftconv_ref(u, k, D, dropout_mask, gelu=True, k_rev=None):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    if k_rev is not None:
        k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
        k_f = k_f + k_rev_f.conj()
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
    
    if len(u.shape) > 3: k_f = k_f.unsqueeze(1)

    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, 'b H -> b H 1')).to(dtype=u.dtype)
    else:
        return out.to(dtype=u.dtype)


@torch.jit.script 
def mul_sum(q, y):
    return (q * y).sum(dim=1)


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
            use_bias=True, # Do we need bias (???)
            use_modulation=True, # Do we need exponential modulation
            num_inner_mlps=2,
            normalized=False,
            **kwargs
        ):
        """
        Implicit long filter with modulation.
        
        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP
        
        Note:
            filter_dropout is not implemented
        """
        super().__init__()
        self.d_model = d_model
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.randn(self.d_model))
        else:
            self.bias = torch.zeros(self.d_model)
        self.fused_fft_conv = fused_fft_conv
        self.dropout = nn.Dropout(dropout)
        
        act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        assert emb_dim % 2 != 0 and emb_dim >= 3, "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.seq_len = seq_len
  
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)

        # uses a variable number of inner linear layers
        self.implicit_filter = nn.Sequential(
            nn.Linear(emb_dim, order),
            act,
        )
        for i in range(num_inner_mlps):
            self.implicit_filter.append(nn.Linear(order, order))
            self.implicit_filter.append(act)
        # final linear layer
        self.implicit_filter.append(nn.Linear(order, d_model, bias=False))

        self.use_modulation = use_modulation
        if self.use_modulation:
            self.modulation = ExponentialModulation(d_model, **kwargs)
        else:
            self.nodulation = None
        
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

        if self.normalized:
            h = h / torch.norm(h, dim=-1, p=1, keepdim=True)

        return h

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None: k = self.filter(L)
        
        # Ensure compatibility with filters that return a tuple 
        k = k[0] if type(k) is tuple else k 
        if bias is None: bias = self.bias

        if self.fused_fft_conv: 
            bias = bias.to(dtype=torch.float32)
            y = fftconv_func(
                x, k, bias, dropout_mask=None, gelu=False, 
                force_fp16_output=torch.is_autocast_enabled()
            )
        else:
            y = fftconv_ref(x, k, bias, dropout_mask=None, gelu=False)

        return y
    
    
class HyenaOperator(nn.Module):
    def __init__(
            self,
            d_model,
            l_max,
            order=2, 
            filter_order=64,
            num_heads=1, 
            inner_factor=1,
            num_blocks=1, 
            fused_bias_fc=False,
            outer_mixing=False,
            dropout=0.0,  
            filter_dropout=0.0, 
            filter_cls='hyena-filter',
            use_context=False, #context parameters
            d_context=None, #context dimension = d_model
            l_context_max=None, #max context length
            context_order=2, #number of blocks of context
            post_order_ffn=False,
            jit_filter=False, 
            short_filter_order=3, 
            activation="id",
            return_state=False,
            **filter_args,
        ):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf
        
        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length. Defaults to None
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            filter_order: (int): Width of the FFN parametrizing the implicit filter. Defaults to 64
            num_heads: (int): Number of heads. Defaults to 1
            inner_factor: (int): Width multiplier. Defaults to 1
            num_blocks: (int): Number of blocks in sequence length. Defaults to 1
            fused_bias_fc: (bool): Whether to use fused bias FC. Defaults to False
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
            post_order_ffn: (bool): Apply a dense layer between steps of the recurrence. Defaults to False
            jit_filter: (bool): Whether JIT the implicit filter function. Defaults to False
            short_filter_order: (int): Length of the explicit input convolutional filter. Defaults to 3
            activation: (str): type of act between kernel output and FF (default identity)
            return_state: (bool): whether to return a state
        """
        super().__init__()
        assert d_model % num_heads == 0, f'Model dimension {d_model} must be divisible by num heads {num_heads}'
        assert l_max % num_blocks == 0, f'Maximum signal length {l_max} must be divisible by block dimension {num_blocks}'
        block_dim = l_max // num_blocks
        head_dim = d_model // num_heads

        if use_context:
            assert d_context==d_model, 'Stacking is not supported'
        
        auto_assign_attrs(
            self, d_model=d_model, order=order, l_max=l_max, num_heads=num_heads, inner_factor=inner_factor, 
            block_dim=block_dim, head_dim=head_dim, filter_order=filter_order, post_order_ffn=post_order_ffn,
            short_filter_order=short_filter_order, num_blocks = num_blocks, filter_dropout=filter_dropout,
            d_context=d_context, use_context=use_context, context_order=context_order, l_context_max=l_context_max,
            jit_filter=jit_filter, outer_mixing=outer_mixing, activation=activation, return_state=return_state,
        )
        self.activation = Activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.setup_projections(fused_bias_fc, inner_factor)
        self.setup_filters(filter_cls, filter_args)


    def setup_projections(self, fused_bias_fc, inner_factor):
        "Initializes input and output projections (over the width dimension)"
        if fused_bias_fc and FusedDense is None:
            raise ImportError('fused_dense is not installed')
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        self.out_proj = linear_cls(self.d_model * inner_factor, self.d_model)
        self.in_proj = linear_cls(self.d_model, (self.order + 1) * self.d_model) 
        if self.post_order_ffn:   
            self.ord_proj_w = nn.Parameter(torch.randn(self.order, self.num_heads, self.num_heads) / math.sqrt(self.head_dim))
        if self.use_context:
            self.context_in_proj = linear_cls(self.d_context, self.context_order * self.d_context)
            self.context_out_proj = linear_cls(self.d_model * inner_factor, self.d_context)
            
            
    def setup_filters(self, filter_cls, filter_args):   
        "Initializes the explicit and implicit filters"
        assert self.order >= 2, f'Order must be at least 2, (got {self.order})'
        total_width = self.d_model * self.inner_factor * (self.order + 1)
        
        self.short_filter = nn.Conv1d(
            in_channels=total_width, 
            out_channels=total_width, 
            kernel_size=self.short_filter_order, 
            groups=total_width, 
            padding=self.short_filter_order - 1
        )
        
        filter_cls = instantiate(registry.layer, filter_cls, partial=True)
                    
        self.filter_fn = filter_cls(
            self.head_dim * self.inner_factor * (self.order - 1), 
            order=self.filter_order, 
            seq_len=self.l_max,
            channels=1, 
            dropout=self.filter_dropout,
            **filter_args
        ) 
        if self.jit_filter: self.filter_fn = torch.jit.script(self.filter_fn, self.L)

        if self.use_context:
            context_width = d_model * self.context_order

            self.context_short_filter = nn.Conv1d(
                in_channels=context_width,
                out_channels=context_width,
                kernel_size=3, # Set this parameter to reduce the number of hyperparameters
                padding=2,
                groups=context_width
            )

            self.context_filter_fn = filter_cls(
                self.head_dim * self.inner_factor * self.context_order, # Model multiplies context and v
                order=self.filter_order,
                seq_len=self.l_max,
                channels=1,
                dropout=self.filter_dropout,
                use_modulation=False,
                w=4 # Set this parameter to reduce the number of hyperparameters
            )
            if self.jit_filter: self.context_filter_fn = torch.jit.script(self.context_filter_fn, self.L)


    def recurrence(self, u , state):
        "Fast inference mode via distilled recurrence"
        raise NotImplementedError("Working on it!")
    
    def forward(self, u, *args, **kwargs):
        if self.use_context:
            u, in_context = u['seq'], u['context']

        l = u.size(-2)
        l_filter = min(l, self.l_max)
        u = self.in_proj(u)
        u = rearrange(u, 'b l d -> b d l')
        
        uc = self.short_filter(u)[...,:l_filter] 
        
        uc = rearrange(uc, 'b (ho v) (z l) -> b ho v z l', 
            z=self.num_blocks, 
            ho=self.num_heads, 
            v=self.head_dim * (self.order + 1)
        )

        *x, v = uc.split(self.d_model, dim=2)
        k = self.filter_fn.filter(l_filter)
        
        # `c` is always 1 by default
        k = rearrange(k, 'c l (v o) -> c o v l', v=self.head_dim, o=self.order - 1)[0]
        
        bias = rearrange(self.filter_fn.bias, '(v o) -> o v', v=self.head_dim, o=self.order - 1)

        if self.use_context:
            l_context = in_context.size(-2)
            # l_context = min(l_context, self.l_context_max)
            context = self.context_in_proj(in_context)
            context = rearrange(context, 'b l d -> b d l')
            if l_filter >= l_context:
                context = nn.functional.pad(context, (0, l_filter-l_context), mode='reflect') # TODO: create different strategies for the context padding
            else:
                print('Warning: context length is greater than sequence length\n')

            context_c = self.context_short_filter(context)[..., :l_filter]

            context_c = rearrange(context_c, 'b (ho v) (z l) -> b ho v z l',
                                  z=self.num_blocks,
                                  ho=self.num_heads,
                                  v=self.head_dim * (self.context_order)
                                )

            prepared_context = context_c.split(self.d_model, dim=2)

            k_context = self.context_filter_fn.filter(l_filter)
            # k_context = nn.functional.normalize(k_context, dim=1)
            k_context = rearrange(k_context, 'c l (v o) -> c o v l', v=self.head_dim, o=self.context_order)[0]

            bias_context = rearrange(self.context_filter_fn.bias, '(v o) -> o v', v=self.head_dim, o=self.context_order)

            x += prepared_context
            k = torch.cat((k, k_context), dim=0)
            bias = torch.cat((bias, bias_context), dim=0)

        for o, x_i in enumerate(reversed(x[1:])):
            if self.outer_mixing:
                v = rearrange(v, 'b h v z l -> b h 1 v z l') 
                v = self.dropout(
                    v * rearrange(x_i, 'b h v z l -> b h v 1 z l') 
                )
                v = v.sum(dim=2)
            else:
                v = self.dropout(v * x_i)

            # the bias term is broadcasted. Last dimension (l) is handled by fftconv
            v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o, None, :, None])
            
            if self.post_order_ffn: 
                w = self.ord_proj_w[o]
                v = mul_sum(
                    rearrange(w, 'h1 h2 -> 1 h1 h2 1 1 1'), rearrange(v, 'b h v z l -> b h 1 v z l')
                )

            # Context Capturing
            if self.use_context and o < self.context_order:
                out_context = v

        y = self.activation(rearrange(v * x[0], 'b h v z l -> b (z l) (h v)', z=self.num_blocks, h=self.num_heads))
        y = self.out_proj(y)

        if self.use_context:
            out_context = self.activation(rearrange(out_context, 'b h v z l -> b (z l) (h v)', z=self.num_blocks, h=self.num_heads))
            out_context = self.context_out_proj(out_context)
        
        if self.return_state:
            return y, None
        if self.use_context:
            y = {'seq': y, 'context': out_context}
            return y
        return y

    @property
    def d_output(self):
        return self.d_model



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
        self.hyena_1 = HyenaOperator(
            d_model=d_model,
            l_max=l_max,
            order=order,
            filter_order=filter_order,
            use_context=use_context,
            d_context=d_context,
            **filter_args
        )
        self.hyena_2 = HyenaOperator(
            d_model=d_model,
            l_max=l_max,
            order=order,
            filter_order=filter_order,
            use_context=True,
            d_context=d_context,
            **filter_args
        )
        self.fc = nn.Linear(d_model, d_context)

    def forward(self, x):
        x = self.hyena_1(x)
        x = self.hyena_2(x)
        x, ctxt = x['seq'], x['context']
        # x = self.hyena_2(x)

        x = self.fc(x)
        return {'seq': x, 'context': ctxt}


if __name__ == "__main__":
    context_dim = 64
    l_max = 2000
    d_model = 64
    dataset_len = 10000
    use_context = True
    num_epochs = 10
    batch_size = 500
    # if not use_context:
    #     l_max *= 2

    import random

    def label_transform(label, length=l_max-10, context_dim=context_dim):

        label = label.item()
        res = []
        for _ in range(length//2):
            wis = random.randint(0, 5)
            res += [label]
            for s in range(wis):
                res += [random.randint(0, context_dim-1)]
            if len(res) > length - length//15:
                res += [random.randint(0, context_dim-1) for _ in range(length-len(res))]
                break
        return res

    # def plot_kernel(model):
    #     import matplotlib.pyplot as plt
    #
    #     k = model.hyena.filter_fn.filter(l_max)[0]  # 1x(oxd) -> (oxd)
    #
    #     k = rearrange(k, 'l (o d) -> o d l', o=model.hyena.order-1)
    #     bias = rearrange(model.hyena.filter_fn.bias, '(o d) -> o d', o=model.hyena.order -1)
    #     print(k.size(), bias.size())
    #
    #     _, idx = k.detach().std(dim=-1).max(dim=-1)
    #     idx = idx[0].item()
    #     print(idx)
    #     for i in range(model.hyena.order -1):
    #         plt.plot(range(l_max), k[i, idx].detach())# + bias[0, idx].detach())
    #     #plt.plot(range(l_max), k[1, idx].detach())# + bias[1, idx].detach())
    #     if model.hyena.use_context:
    #         k_context = model.hyena.context_filter_fn.filter(l_max)[0]
    #         k_context = rearrange(k_context, 'l (o d) -> o d l', o=2)
    #         bias_context = rearrange(model.hyena.context_filter_fn.bias, '(o d) -> o d', o=2)
    #         plt.plot(range(l_max), k_context[0, idx].detach())# + bias_context[0, idx].detach())
    #         plt.plot(range(l_max), k_context[1, idx].detach())# + bias_context[1, idx].detach())
    #     plt.show()


    model = MyModel(
        d_model=d_model,
        l_max=l_max if use_context else l_max*2,
        order=3,
        filter_order=16,
        use_context=use_context,
        d_context=context_dim,
        w=10,
        emb_dim=3
    )
    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

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
        # if epoch%1==0:
        #     plot_kernel(model)
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

            x = {'seq': x, 'context': ctxt}

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(x)['seq'][:, -1, :]
            #outputs = model(ctxt, x)[:, -1, :]
            # print(outputs.size())
            # print(outputs[:, -1, :].size())
            loss = criterion(outputs, lbls)
            running_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()


            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += lbls.size(0)
            correct += (predicted == lbls).sum().item()
            if i%1000==0:
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
                    y_pred = model(
                            {'seq': torch.cat(t_sequences, dim=0),
                              'context': torch.cat(t_context, dim=0)}
                              )['seq'][:, -1, :]

                    t_loss = criterion(y_pred, t_labels)
                    y_pred = torch.max(
                        y_pred,
                        dim=1
                    )[1]
                    t_correct = (y_pred == t_labels).sum().item()
                    print(f'after {i} sequences Acc: {t_correct/100 :.4f} Loss:{t_loss.item() :.4f}')

                    #loss_val = criterion(y_pred_val.squeeze(), y_val.float())

                model.train()

        # Compute and print the epoch loss and accuracy
        epoch_loss = running_loss / (len(sequences) / batch_size)
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}")
        lr_scheduler.step()

    # plot_kernel(model)