from typing import Optional, Protocol, Tuple

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW


class Attention(Protocol):
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Implementation of attention. Should return tuple of (feature, attention_map).
        """


def plain_attention(
    query: torch.Tensor,
    key_value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    attention_dropout: float = 0.0,
    training: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Args:
        query: (batch, out_seq_len, dim)
        key_value: (batch, in_seq_len, 2, dim)
        mask: (batch, out_seq_len, in_seq_len)
        attention_dropout: dropout probability
        training: whether in training mode

    Returns:
        Tuple[Tensor, Tensor]: (feature, attention_map)
        where:
            feature: (batch, out_seq_len, dim)
            attention_map: (batch, heads, out_seq_len, in_seq_len)
    """
    key, value = key_value.unbind(2)

    keyT = key.permute(0, 2, 3, 1)  # transpose to (batch, heads, dim, in_seq_len)
    value = value.transpose(1, 2)  # transpose to (batch, heads, in_seq_len, dim)
    query = query.transpose(1, 2)  # transpose to (batch, heads, out_seq_len, dim)

    softmax_scale = query.shape[-1] ** (-0.5)
    dots = torch.matmul(query * softmax_scale, keyT)
    if mask is not None:
        assert (
            mask.shape[-2:] == dots.shape[-2:]
        ), f"Mask shape {mask.shape} does not match attention shape {dots.shape}"
        inv_mask = (
            (~mask).unsqueeze(-3).expand_as(dots)
        )  # pylint: disable=invalid-unary-operand-type
        dots.masked_fill_(inv_mask, float("-inf"))

    attn = dots.softmax(dim=-1, dtype=torch.float).to(
        value.dtype
    )  # (batch, heads, out_seq_len, in_seq_len)
    if attention_dropout > 0:
        attn = F.dropout(attn, p=attention_dropout, training=training)

    y = torch.matmul(attn, value).transpose(
        1, 2
    )  # transpose to (batch, seq_len, heads, dim)
    return y, attn


class PlainAttention(nn.Module):
    """
    Attention module from original Transformer paper.
    """

    def __init__(
        self,
        model_dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        head_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        context_dim = model_dim if context_dim is None else context_dim
        if head_dim is None:
            assert (
                model_dim % num_heads == 0
            ), f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
            head_dim = model_dim // num_heads
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.to_q = nn.Linear(model_dim, head_dim * num_heads, bias=False)
        self.to_kv = nn.Linear(context_dim, head_dim * num_heads * 2, bias=False)
        self.to_out = nn.Linear(head_dim * num_heads, model_dim)

    def forward(
        self, x, context=None, mask: Optional[torch.Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        :param x: [batch, seq_len, model_dim]
        :param context: [batch, context_len, context_dim]
        :param mask: [batch, seq_len, context_len]
        """

        context = x if context is None else context
        query = rearrange(
            self.to_q(x),
            "batch seq (head feature) -> batch seq head feature",
            head=self.num_heads,
        )
        key_value = rearrange(
            self.to_kv(context),
            "batch seq (n head feature) -> batch seq n head feature",
            head=self.num_heads,
            n=2,
        )
        y, attn = plain_attention(
            query=query,
            key_value=key_value,
            mask=mask,
            attention_dropout=self.attention_dropout,
            training=self.training,
        )
        y = self.to_out(y.flatten(-2))
        return y, attn


class PyTorchAttention(nn.Module):
    """
    Attention module using the PyTorch MultiheadAttention module.
    Currently slower and less flexible than PlainAttention, but
    this hopefully improves as we upgrade PyTorch.
    """

    def __init__(
        self,
        model_dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        context_dim = model_dim if context_dim is None else context_dim
        self.mha = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
            kdim=context_dim,
            vdim=context_dim,
        )

    def forward(
        self, x, context=None, mask: Optional[torch.Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        :param x: [batch, seq_len, model_dim]
        :param context: [batch, context_len, context_dim]
        :param mask: [batch, seq_len, context_len]
        """
        context = x if context is None else context
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            # We invert the mask here, because for torch True means *empty*,
            mask = repeat(
                # pylint: disable=invalid-unary-operand-type
                ~mask,
                "batch query key -> (batch head) query key",
                head=self.mha.num_heads,
            )
        y, attn = self.mha(
            query=x,
            key=context,
            value=context,
            attn_mask=mask,
            need_weights=True,
            average_attn_weights=False,
        )
        return y, attn


############################################################################################################
############################################################################################################
############################################################################################################
# What follows are various attention based models.                                                         #
############################################################################################################
############################################################################################################
############################################################################################################


def make_ffn(dim, mult=4):
    return nn.Sequential(
        nn.Linear(dim, dim * mult, bias=False),
        nn.GELU(),
        nn.Linear(dim * mult, dim, bias=False),
    )


def generate_square_subsequent_mask(
    num_queries: int, num_keys: int, device: str = "cpu", diagonal: int = 0
) -> torch.Tensor:
    """Generate the attention mask for causal decoding"""
    mask = torch.ones(num_queries, num_keys, dtype=torch.bool, device=device)
    return torch.tril(
        mask, diagonal=diagonal + max(num_keys - num_queries, 0), out=mask
    )


class TransformerBlock(nn.Module):
    """
    A transformer block with pre-normalization.
    """

    def __init__(
        self,
        model_dim: int,
        attention: Attention,
        context_dim: Optional[int] = None,
        extra_context_norm: bool = False,
    ):
        super().__init__()
        context_dim = model_dim if context_dim is None else context_dim
        self.attention = attention
        self.ff = make_ffn(model_dim)
        self.pre_norm1 = nn.LayerNorm(model_dim)
        self.pre_norm2 = nn.LayerNorm(context_dim) if extra_context_norm else None
        self.pre_norm3 = nn.LayerNorm(model_dim)

    def forward(self, x, context=None, mask=None) -> Tuple[Tensor, Tensor]:
        context = x if context is None else context
        if self.pre_norm2 is not None:
            y, attn = self.attention.forward(
                self.pre_norm1(x), context=self.pre_norm2(context), mask=mask
            )
        elif x is not context:
            y, attn = self.attention.forward(
                self.pre_norm1(x), context=self.pre_norm1(context), mask=mask
            )
        else:
            y, attn = self.attention.forward(self.pre_norm1(x), mask=mask)
        x = x + y
        x = x + self.ff(self.pre_norm3(x))
        return x, attn


class Transformer(nn.Module):
    """
    A self attention transformer like GPT or BERT
    """

    def __init__(
        self,
        model_dim: int,
        depth: int,
        heads: int = 8,
        attention_dropout: float = 0.0,
        causal=True,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.causal = causal
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    model_dim,
                    PlainAttention(model_dim, model_dim, heads, attention_dropout),
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(
        self, token: Tensor, state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            token: [batch, model_dim] or [batch, seq_len, model_dim]
            state: [batch, seq_len, model_dim]
        Returns:
            [batch, model_dim], new_state
        """
        assert token.shape[-1] == self.model_dim
        x = token.unsqueeze(-2) if token.dim() == 2 else token

        # Preserve causality if causal=True
        num_queries = x.shape[-2]
        num_keys = x.shape[-2] + (state.shape[-3] if state is not None else 0)
        mask = (
            generate_square_subsequent_mask(num_queries, num_keys, device=x.device)
            if self.causal
            else None
        )

        # Forward through blocks
        new_state = []
        attention_maps = torch.zeros(
            (
                x.shape[0],
                x.shape[1] + state.shape[1] if state is not None else x.shape[1],
            )
        )
        for i, block in enumerate(self.blocks):
            # The context consists of both previous and new state
            context = torch.cat((state[:, :, i, :], x), 1) if state is not None else x
            new_state.append(context)
            x, attn = block(x, context=context, mask=mask)
            attention_maps += attn.sum(dim=(-2, -3))

        x = self.output_norm(x)
        x = x.squeeze(1) if token.dim() == 2 else x
        return x, torch.stack(new_state, 2), attention_maps


class CrossPerceiver(nn.Module):
    """
    A residual MLP interleaved with cross attention.
    """

    def __init__(self, model_dim: int, context_dim: int, num_blocks=5, num_heads=8):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    model_dim=model_dim,
                    attention=PlainAttention(
                        model_dim=model_dim,
                        context_dim=context_dim,
                        num_heads=num_heads,
                    ),
                    context_dim=context_dim,
                    extra_context_norm=True,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, x, context, context_mask):
        x = x.unsqueeze(-2)  # [b, 1, d]
        context_mask = context_mask.unsqueeze(-2)  # [b, 1, c]

        attention_maps = torch.zeros((context.shape[0], context.shape[1]))
        for block in self.blocks:
            x, attn = block(x, context=context, mask=context_mask)
            attention_maps += attn.sum(dim=(-2, -3))

        x = self.output_norm(x).squeeze(-2)  # [b, d]
        # [b(including sequence!), layers, heads, context]
        return x, attention_maps


class Perceiver(nn.Module):
    """
    PERCEIVER IO: A GENERAL ARCHITECTURE FOR STRUCTURED INPUTS & OUTPUTS
    https://arxiv.org/abs/2107.14795
    """

    def __init__(
        self,
        model_dim: int,
        context_dim: int,
        num_latents: int,
        num_blocks=5,
        num_heads=8,
        num_queries=1,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.num_queries = num_queries
        self.input_embedding = nn.Parameter(torch.empty((num_latents, model_dim)))
        self.output_embedding = nn.Parameter(torch.empty((num_queries, model_dim)))
        self.input_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.input_ff = nn.Sequential(nn.LayerNorm(model_dim), make_ffn(model_dim))
        self.input_block = TransformerBlock(
            model_dim=model_dim,
            attention=PlainAttention(
                model_dim=model_dim, context_dim=context_dim, num_heads=1
            ),
            context_dim=context_dim,
            extra_context_norm=True,
        )
        self.output_block = TransformerBlock(
            model_dim=model_dim,
            attention=PlainAttention(model_dim=model_dim, num_heads=1),
            extra_context_norm=True,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    model_dim,
                    PlainAttention(model_dim=model_dim, num_heads=num_heads),
                    context_dim,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self._init_parameters()

    def _init_parameters(self):
        self.input_embedding.data.normal_(0.0, 0.02)
        self.output_embedding.data.normal_(0.0, 0.02)

    def forward(self, x, context, context_mask=None):
        context_mask = (
            context_mask.unsqueeze(1).expand(-1, self.num_latents, -1)
            if context_mask is not None
            else None
        )

        latent = self.input_embedding.expand(context.shape[0], -1, -1)
        if x is not None:
            latent = (
                latent + self.input_proj(x).unsqueeze(1).expand_as(latent)
                if x.dim() == 2
                else latent + self.input_proj(x)
            )
            latent = latent + self.input_ff(latent)

        latent, input_attn = self.input_block(
            latent, context=context, mask=context_mask
        )

        attention_maps = latent.new_zeros(
            (context.shape[0], context.shape[1]), requires_grad=False
        )
        for block in self.blocks:
            latent, attn = block(latent)
            # [batch, heads, latents, context] => [batch, context]
            attention_maps += (attn @ input_attn).sum(dim=(-2, -3))

        y = self.output_embedding.expand(context.shape[0], -1, -1)
        y, _ = self.output_block(y, context=latent)
        y = self.output_norm(y)
        return y, attention_maps


def configure_optimiser(
    module: nn.Module, lr: float, weight_decay: float, betas=(0.9, 0.999), eps=1e-5
) -> AdamW:
    """
    Separate parameters into two groups: regularized and non-regularized, then return optimizer.
    """
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Linear, nn.MultiheadAttention)
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
    for mn, m in module.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
            if pn.endswith("bias") or pn.endswith("embedding") or pn.endswith("alpha"):
                # all biases won't be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif isinstance(m, (nn.GRU, nn.LSTM, nn.GRUCell)):
                # all recurrent weights will not be decayed
                no_decay.add(fpn)

    # Validate that we considered every parameter
    param_dict = {pn: p for pn, p in module.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay

    if len(inter_params) > 0:
        raise ValueError(
            f"Parameters {inter_params} are in both decay/no_decay groups."
        )

    if union_params != set(param_dict.keys()):
        raise ValueError(
            f"Parameters {set(param_dict.keys()).difference(union_params)} were not separated into either decay/no_decay set."
        )

    return AdamW(
        params=[
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ],
        lr=lr,
        betas=betas,
        eps=eps,
        foreach=True,
    )
