import torch
from torch import nn, einsum
import torch.nn.functional as F

from collections import namedtuple
from functools import wraps
from packaging import version
from einops import rearrange, repeat


"""
Config 是一个命名元组，用于存储配置参数：
    - enable_flash: 是否启用 Flash 功能
    - enable_math: 是否启用数学功能
    - enable_mem_efficient: 是否启用内存优化功能
"""
Config = namedtuple('Config', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


def exists(val):
    """
    检查值是否存在（即不为 None）。

    Args:
        val: 需要检查的值

    Returns:
        bool: 如果值存在则返回 True，否则返回 False
    """
    return val is not None


def once(fn):
    """
    装饰器，确保被装饰的函数只执行一次。

    Args:
        fn (function): 需要限制只执行一次的函数

    Returns:
        function: 包装后的函数
    """
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)


class Attend(nn.Module):
    """
    注意力机制模块，支持 Flash 注意力、内存高效注意力和数学优化的注意力。

    Args:
        dropout (float, optional): Dropout 概率。默认为 0。
        causal (bool, optional): 是否使用因果掩码。默认为 False。
        flash (bool, optional): 是否尝试使用 Flash 注意力。默认为 False。
    """
    def __init__(
        self,
        dropout = 0.,
        causal = False,
        flash = False
    ):
        super().__init__()
        # Dropout 概率
        self.dropout = dropout
        # Dropout 层
        self.attn_dropout = nn.Dropout(dropout)

        # 是否使用因果掩码
        self.causal = causal
        # 注册缓冲区用于存储掩码
        self.register_buffer("mask", None, persistent=False)

        # 是否尝试使用 Flash 注意力
        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu
        # 确定在 CUDA 和 CPU 上的高效注意力配置

        # CPU 配置：使用 Flash、内存高效和数学优化
        self.cpu_config = Config(True, True, True)
        # CUDA 配置初始化为 None
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            # 如果没有可用的 CUDA 或不尝试使用 Flash 注意力，则返回
            return

        # 获取 CUDA 设备属性
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        # 如果是 A100 GPU，则使用 Flash 注意力
        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = Config(True, False, False)
        else:
            # 如果不是 A100 GPU，则使用内存高效和数学优化的注意力
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = Config(False, True, True)

    def flash_attn(self, q, k, v, mask = None):
        """
        使用 Flash 注意力机制计算注意力输出。

        Args:
            q (Tensor): 查询张量，形状为 (batch_size, heads, query_length, head_dim)。
            k (Tensor): 键张量，形状为 (batch_size, heads, key_length, head_dim)。
            v (Tensor): 值张量，形状为 (batch_size, heads, key_length, head_dim)。
            mask (Optional[Tensor], optional): 掩码张量，形状为 (batch_size, query_length, key_length)。默认为 None。

        Returns:
            Tensor: 注意力输出，形状为 (batch_size, heads, query_length, head_dim)。
        """
        # 获取张量的形状信息
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        # 重复键和值张量以匹配多头注意力的维度
        k = repeat(k, 'b ... -> b h ...', h = heads)
        v = repeat(v, 'b ... -> b h ...', h = heads)

        # 是否使用因果掩码
        causal = self.causal

        if exists(mask):
            # 重塑掩码张量的形状以匹配多头注意力的维度
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)

            if causal:
                # 创建因果掩码
                causal_mask = torch.ones((q_len, k_len), device = q.device, dtype = torch.bool).triu(k_len - q_len + 1)
                mask = mask & ~causal_mask    
                # 取消因果掩码标志            
                causal = False 

        # 选择注意力机制的配置
        config = self.cuda_config if is_cuda else self.cpu_config

        # 使用 Scaled Dot Product Attention 进行计算
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = causal
            )

        return out

    def forward(self, q, k, v, mask = None, attn_bias = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        """
        前向传播方法，实现多头自注意力机制。

        参数说明（使用爱因斯坦求和约定表示）：
        - b: 批次大小
        - h: 注意力头数
        - n, i, j: 序列长度（基础序列长度，源序列长度，目标序列长度）
        - d: 特征维度

        Args:
            q (torch.Tensor): 查询张量，形状为 (b, h, n, d)
            k (torch.Tensor): 键张量，形状为 (b, h, j, d)
            v (torch.Tensor): 值张量，形状为 (b, h, j, d)
            mask (torch.Tensor, optional): 掩码张量，用于掩盖特定的元素。默认为 None。
            attn_bias (torch.Tensor, optional): 注意力偏置张量，用于添加额外的注意力权重。默认为 None。

        Returns:
            torch.Tensor: 输出张量，形状为 (b, h, n, d)
        """
        # 获取序列长度和设备信息
        n, device = q.shape[-2], q.device

        # 计算缩放因子，通常为 sqrt(d_k)
        scale = q.shape[-1] ** -0.5

        if self.flash:
            assert not exists(attn_bias), 'attention bias not supported for flash attention'
            # 使用 Flash Attention 进行计算
            return self.flash_attn(q, k, v, mask = mask)

        # similarity
        # 计算相似度矩阵
        sim = einsum("b h i d, b j d -> b h i j", q, k) * scale

        # attention bias
        # 添加注意力偏置
        if exists(attn_bias):
            sim = sim + attn_bias # 加上偏置项

        # key padding mask
        # 应用键掩码（key padding mask）
        if exists(mask):
            # 重塑掩码形状以匹配相似度矩阵
            mask = rearrange(mask, 'b j -> b 1 1 j')
            # 将不需要注意的位置设为负无穷大
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal mask
        # 应用因果掩码（causal mask）
        if self.causal:
            # 获取相似度矩阵的形状
            i, j = sim.shape[-2:]
            # 创建上三角掩码
            causal_mask = torch.ones((i, j), device = sim.device, dtype = torch.bool).triu(j - i + 1)
            # 将未来的位置设为负无穷大
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention
        # 计算注意力权重
        # 对相似度矩阵进行 softmax 归一化
        attn = sim.softmax(dim=-1)
        # 应用 Dropout
        attn = self.attn_dropout(attn)

        # aggregate values
        # 计算最终的输出值
        out = einsum("b h i j, b j d -> b h i d", attn, v)

        return out
