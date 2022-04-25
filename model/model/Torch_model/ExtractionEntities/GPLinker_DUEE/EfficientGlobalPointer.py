import torch
from torch.nn import Module
from torch import nn
import math
import numpy as np

class SinusoidalPositionEmbedding(Module):
    """定义Sin-Cos位置Embedding
    """
    def __init__(
            self, output_dim, merge_mode='add', custom_position_ids=False):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        input_shape = inputs.shape
        batch_size, seq_len = input_shape[0], input_shape[1]
        position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)


def relative_position_encoding(depth, max_length=512, max_relative_position=127):
    vocab_size = max_relative_position * 2 + 1
    range_vec = torch.arange(max_length)
    range_mat = range_vec.repeat(max_length).view(max_length, max_length)
    distance_mat = range_mat - torch.t(range_mat)
    distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
    final_mat = distance_mat_clipped + max_relative_position

    embeddings_table = torch.zeros(vocab_size, depth)
    position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, depth, 2).float() * (-math.log(10000.0) / depth))
    embeddings_table[:, 0::2] = torch.sin(position * div_term)
    embeddings_table[:, 1::2] = torch.cos(position * div_term)
    embeddings_table = embeddings_table.unsqueeze(0).transpose(0, 1).squeeze(1)

    flat_relative_positions_matrix = final_mat.view(-1)
    one_hot_relative_positions_matrix = torch.nn.functional.one_hot(flat_relative_positions_matrix,
                                                                    num_classes=vocab_size).float()
    positions_encoding = torch.matmul(one_hot_relative_positions_matrix, embeddings_table)
    my_shape = list(final_mat.size())
    my_shape.append(depth)
    positions_encoding = positions_encoding.view(my_shape)
    return positions_encoding


def sequence_masking(x, mask, value='-inf', axis=None):
    if mask is None:
        return x
    else:
        if value == '-inf':
            value = -1e12
        elif value == 'inf':
            value = 1e12
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = torch.unsqueeze(mask, 1)
        for _ in range(x.ndim - mask.ndim):
            mask = torch.unsqueeze(mask, mask.ndim)
        return x * mask + value * (1 - mask)


def add_mask_tril(logits, mask):
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 2)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 1)
    # 排除下三角
    mask = torch.tril(torch.ones_like(logits), diagonal=-1)
    logits = logits - mask * 1e12
    return logits


def align(tensor, axes, ndim=None):
    """重新对齐tensor（批量版expand_dims）
    axes：原来的第i维对齐新tensor的第axes[i]维；
    ndim：新tensor的维度。
    """
    assert len(axes) == len(tensor.shape)
    assert ndim or min(axes) >= 0
    ndim = ndim or max(axes) + 1
    indices = [None] * ndim
    for i in axes:
        indices[i] = slice(None)
    return tensor[indices]


def apply_rotary_position_embeddings(sinusoidal, qw, kw):
    ndim = qw.ndim
    sinusoidal = align(sinusoidal, [0, 1, -1], ndim)
    cos_pos = sinusoidal[..., 1::2].repeat_interleave(2,dim=-1)
    sin_pos = sinusoidal[..., ::2].repeat_interleave(2,dim=-1)
    qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], ndim)
    qw2 = torch.reshape(qw2, qw.shape)
    qw = qw * cos_pos + qw2 * sin_pos
    kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], ndim)
    kw2 = torch.reshape(kw2, kw.shape)
    kw = kw * cos_pos + kw2 * sin_pos
    return qw, kw


class GlobalPointer(Module):

    def __init__(self, heads, head_size, hidden_size, RoPE=True):
        super(GlobalPointer, self).__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.dense = nn.Linear(hidden_size, self.head_size * self.heads * 2)

    def forward(self, inputs, mask=None):
        B, N, D = inputs.shape
        inputs = self.dense(inputs) # B, N, 2*head_size*heads_num
        inputs = inputs.reshape(B, N, self.heads, self.head_size * 2) # B, N, heads_num, 2*head_size
        qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:] # B, N, heads_num, head_size
        # 分出qw和kw
        # RoPE编码
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
            qw,kw = apply_rotary_position_embeddings(pos, qw, kw)
        # 计算内积
        # B, N, heads_num, head_size
        logits = torch.einsum('bmhd , bnhd -> bhmn', qw, kw)
        # 排除padding 排除下三角
        logits = add_mask_tril(logits, mask)
        # scale返回
        return logits / self.head_size ** 0.5


class EfficientGlobalPointer(GlobalPointer):

    def __init__(self, heads, head_size, hidden_size, RoPE=True):
        super(EfficientGlobalPointer, self).__init__(heads, head_size, hidden_size, RoPE)
        self.p_dense = nn.Linear(hidden_size, self.head_size * 2)
        self.q_dense = nn.Linear(self.head_size * 2, self.heads * 2)

    def forward(self, inputs, mask=None):
        # 输入变换
        inputs = self.p_dense(inputs) # b n head_size*2
        qw, kw = inputs[..., ::2], inputs[..., 1::2] # b n head_size
        # RoPE编码
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
            qw, kw = apply_rotary_position_embeddings(pos, qw, kw)
        # 计算内积 计算实体得分
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.head_size**0.5 # b m n

        '''
        wa * [hi;hj] = wa_1 * h1 + wa_2 * h2
        (head , 2D) * (2D, n, n)  = (head, n , n) 
        (head, D) * (D, n ,n) = (head, n, n)
        
        (n , D) * (D , 2 * head) = (n , 2 * head)
         = [(n , D) * wa_1(D , head) ;(n , D) * wa_2(D , head)]
         
        (head , n , 1) (wa_1 * h) + (head , 1, n) (wa_2 * h) 
        = (head, n , n) = wa_1 * h1 + wa_2 * h2
        
        self.q_dense = nn.Linear(self.head_size * 2 * 2, self.heads)
        x1 = torch.unsqueeze(inputs, 1)
        x2 = torch.unsqueeze(inputs, 2)
        x1 = x1.repeat(1, N, 1, 1)
        x2 = x2.repeat(1, 1, N, 1)
        concat_x = torch.cat([x2, x1], dim=-1) B N N 2D
        output  = self.q_dense(concat_x) B N N H
        '''

        bias = torch.einsum('bnh->bhn',self.q_dense(inputs))/2
        #self.q_dense(inputs).permute(0, 2, 1) / 2

        # b heads*2 n
        # 利用加法传播 b 1 m n  + b heads 1 n   + b heads m 1 ——> b head m n
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]

        # 排除padding 与 三角 mask
        logits = sequence_masking(logits, mask, '-inf', 2)
        logits = sequence_masking(logits, mask, '-inf', 3)
        
        tri_mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = logits - tri_mask * 1e12
        return logits
        # 返回最终结果