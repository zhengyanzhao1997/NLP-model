import torch
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch import nn
import math

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
    if mask.dtype != logits.dtype:
        mask = mask.type(logits.dtype)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 2)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 1)
    # 排除下三角
    mask = torch.tril(torch.ones_like(logits), diagonal=-1)
    logits = logits - mask * 1e12
    return logits

class GlobalPointer(Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """
    def __init__(self, heads, head_size,hidden_size,RoPE=True):
        super(GlobalPointer, self).__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.dense = nn.Linear(hidden_size,self.head_size * self.heads * 2)

    def forward(self, inputs, mask=None):
        inputs = self.dense(inputs)
        inputs = torch.split(inputs, self.head_size * 2 , dim=-1)
        # 按照-1这个维度去分，每块包含x个小块
        inputs = torch.stack(inputs, dim=-2)
        #沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状
        qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]
        #分出qw和kw
        # RoPE编码
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
            cos_pos = pos[..., None, 1::2].repeat(1,1,1,2)
            sin_pos = pos[..., None, ::2].repeat(1,1,1,2)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 4)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 4)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # 计算内积
        logits = torch.einsum('bmhd , bnhd -> bhmn', qw, kw)
        # 排除padding 排除下三角
        logits = add_mask_tril(logits,mask)
        # scale返回
        return logits / self.head_size ** 0.5


class MutiHeadSelection(Module):

    def __init__(self,hidden_size,c_size,abPosition = False,rePosition=False, maxlen=None,max_relative=None):
        super(MutiHeadSelection, self).__init__()
        self.hidden_size = hidden_size
        self.c_size = c_size
        self.abPosition = abPosition
        self.rePosition = rePosition
        self.Wh = nn.Linear(hidden_size * 2,self.hidden_size)
        self.Wo = nn.Linear(self.hidden_size,self.c_size)
        if self.rePosition:
            self.relative_positions_encoding = relative_position_encoding(max_length=maxlen,
                                                                     depth=2*hidden_size,
                                                                     max_relative_position=max_relative)
    def forward(self, inputs, mask=None):
        input_length = inputs.shape[1]
        batch_size = inputs.shape[0]
        if self.abPosition:
            # 由于为加性拼接，我们无法使用RoPE,因此这里直接使用绝对位置编码
            inputs = SinusoidalPositionEmbedding(self.hidden_size, 'add')(inputs)
        x1 = torch.unsqueeze(inputs, 1)
        x2 = torch.unsqueeze(inputs, 2)
        x1 = x1.repeat(1, input_length, 1, 1)
        x2 = x2.repeat(1, 1, input_length, 1)
        concat_x = torch.cat([x2, x1], dim=-1)
        if self.rePosition:
            relations_keys = self.relative_positions_encoding[:input_length, :input_length, :].to(inputs.device)
            concat_x += relations_keys
        hij = torch.tanh(self.Wh(concat_x))
        logits = self.Wo(hij)
        logits = logits.permute(0,3,1,2)
        logits = add_mask_tril(logits, mask)
        return logits


class Biaffine(Module):

    def __init__(self, in_size, out_size, Position = False):
        super(Biaffine, self).__init__()
        self.out_size = out_size
        self.weight1 = Parameter(torch.Tensor(in_size, out_size, in_size))
        self.weight2 = Parameter(torch.Tensor(2 * in_size + 1, out_size))
        self.Position = Position
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight1,a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.weight2,a=math.sqrt(5))
    
    def forward(self, inputs, mask = None):
        input_length = inputs.shape[1]
        hidden_size = inputs.shape[-1]
        if self.Position:
            # 由于为加性拼接，我们无法使用RoPE,因此这里直接使用绝对位置编码
            inputs = SinusoidalPositionEmbedding(hidden_size, 'add')(inputs)
        x1 = torch.unsqueeze(inputs, 1)
        x2 = torch.unsqueeze(inputs, 2)
        x1 = x1.repeat(1, input_length, 1, 1)
        x2 = x2.repeat(1, 1, input_length, 1)
        concat_x = torch.cat([x2, x1], dim=-1)
        concat_x = torch.cat([concat_x, torch.ones_like(concat_x[..., :1])],dim=-1)
        # bxi,oij,byj->boxy
        logits_1 = torch.einsum('bxi,ioj,byj -> bxyo', inputs, self.weight1, inputs)
        logits_2 = torch.einsum('bijy,yo -> bijo', concat_x, self.weight2)
        logits = logits_1 + logits_2
        logits = logits.permute(0,3,1,2)
        logits = add_mask_tril(logits, mask)
        return logits


class TxMutihead(Module):

    def __init__(self,hidden_size,c_size,abPosition = False,rePosition=False, maxlen=None,max_relative=None):
        super(TxMutihead, self).__init__()
        self.hidden_size = hidden_size
        self.c_size = c_size
        self.abPosition = abPosition
        self.rePosition = rePosition
        self.Wh = nn.Linear(hidden_size * 4, self.hidden_size)
        self.Wo = nn.Linear(self.hidden_size,self.c_size)
        if self.rePosition:
            self.relative_positions_encoding = relative_position_encoding(max_length=maxlen,
                                                                     depth=4*hidden_size,
                                                                     max_relative_position=max_relative)
    def forward(self, inputs, mask=None):
        input_length = inputs.shape[1]
        batch_size = inputs.shape[0]
        if self.abPosition:
            # 由于为加性拼接，我们无法使用RoPE,因此这里直接使用绝对位置编码
            inputs = SinusoidalPositionEmbedding(self.hidden_size, 'add')(inputs)
        x1 = torch.unsqueeze(inputs, 1)
        x2 = torch.unsqueeze(inputs, 2)
        x1 = x1.repeat(1, input_length, 1, 1)
        x2 = x2.repeat(1, 1, input_length, 1)
        concat_x = torch.cat([x2, x1,x2-x1,x2.mul(x1)], dim=-1)
        if self.rePosition:
            relations_keys = self.relative_positions_encoding[:input_length, :input_length, :].to(inputs.device)
            concat_x += relations_keys
        hij = torch.tanh(self.Wh(concat_x))
        logits = self.Wo(hij)
        logits = logits.permute(0,3,1,2)
        logits = add_mask_tril(logits, mask)
        return logits
