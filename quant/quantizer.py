import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def lp_loss(pred, tgt, p=2.0, reduction="none"):
    """
    loss function measured in L_p Norm
    """
    if reduction == "none":
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


class UniformQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """

    def __init__(self, n_bits: int = 8, channel_wise: bool = False):
        super(UniformQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, "bitwidth not supported"
        self.n_bits = n_bits
        self.n_levels = 2**self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise

    def __repr__(self):
        s = super(UniformQuantizer, self).__repr__()
        s = (
            "("
            + s
            + " inited={}, channel_wise={})".format(self.inited, self.channel_wise)
        )
        return s

    def forward(self, x: torch.Tensor):

        if self.inited is False:
            self.delta, self.zero_point = self.init_quantization_scale(
                x, self.channel_wise
            )
            self.inited = True

        # start quantization
        x_int = torch.round(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=-1)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(
                        x_clone[:, :, c], channel_wise=False
                    )
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(
                        x_clone[c], channel_wise=False
                    )
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
            else:
                raise NotImplementedError
        else:
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()
            best_score = 1e10
            for pct in [0.999, 0.9999, 0.99999]:
                try:
                    new_max = torch.quantile(x_clone.reshape(-1), pct)
                    new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                except:
                    new_max = torch.tensor(
                        np.percentile(x_clone.reshape(-1).cpu(), pct * 100),
                        device=x_clone.device,
                        dtype=torch.float32,
                    )
                    new_min = torch.tensor(
                        np.percentile(x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                        device=x_clone.device,
                        dtype=torch.float32,
                    )
                x_q = self.quantize(x_clone, new_max, new_min)
                score = lp_loss(x_clone, x_q, p=2, reduction="all")
                if score < best_score:
                    best_score = score
                    delta = (new_max - new_min) / (2**self.n_bits - 1)
                    zero_point = (-new_min / delta).round()

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2**self.n_bits - 1)
        zero_point = (-min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q


class LogSqrt2Quantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """

    def __init__(
        self, n_bits: int = 8, channel_wise: bool = False, log_quant_scheme="Sqrt2_17"
    ):
        super(LogSqrt2Quantizer, self).__init__()
        assert 2 <= n_bits <= 8, "bitwidth not supported"
        self.n_bits = n_bits
        self.n_levels = 2**self.n_bits
        self.delta = None
        self.inited = False
        self.channel_wise = channel_wise
        self.log_quant_scheme = log_quant_scheme

        """when using 4Bit INT Log2 Quantization"""
        if self.log_quant_scheme == "BitLog2_Single_16":
            self.int_max = 32768
        elif self.log_quant_scheme == "BitLog2_Single_17":
            self.int_max = 65536
        elif self.log_quant_scheme == "BitLog2_Half_16":
            self.int_max = 256
        elif self.log_quant_scheme == "BitLog2_Half_17":
            self.int_max = 384
        else:
            self.int_max = None

    def bitLog2Single(self, x):
        # 1. mask 0
        zeromask = x == 0
        # 2. get bit length - 1 (same as the bit length for representation)
        for i in torch.arange(0, 16):
            x = torch.where(x.bitwise_right_shift(i) == 1, i, x)
        # 3. -inf
        x[zeromask] = -99999
        x_q = x.clone().detach()
        # 4. dequantize
        x_dq = 2**x
        x_dq[zeromask] = 0

        return x_q, x_dq

    def bitLog2Half(self, x):
        # 1. get log2(x)
        x_q, _ = self.bitLog2Single(x)

        def bitLog2Half_quant(x, x_q):
            # 2. get the mask for 1 (1 is 0.5)
            one_mask_half = x == 1
            # 3. get the half value
            x_temp = torch.where(
                x.bitwise_right_shift(x_q - 1).bitwise_and(1) == 1, 5, 0
            )
            # 4. 1 is 0.5
            x_temp[one_mask_half] = 5
            # 5. get the quantized value (int_part * 10 + frac_part)
            x_q_half = x_q * 10 + x_temp
            return x_q_half

        def bitLog2Half_dequant(x_q_half):
            # 1. get the mask for 0
            zero_mask = x_q_half == 0
            # 2. get the int part and frac part
            int_part = x_q_half // 10
            frac_part = x_q_half % 10 // 5
            _one = torch.ones_like(int_part)
            # 3. get the dequantized value
            int_num = _one.bitwise_left_shift(int_part)
            frac_num = frac_part * _one.bitwise_left_shift(int_part - 1)
            x_dq_half = int_num + frac_num
            x_dq_half[zero_mask] = 0

            return x_dq_half

        x_q_half = bitLog2Half_quant(x, x_q)
        x_dq_half = bitLog2Half_dequant(x_q_half)

        return x_q_half, x_dq_half

    def forward_logquant(self, x: torch.Tensor):
        if "BitLog2" in self.log_quant_scheme:
            x_int = torch.floor(x * self.int_max).to(torch.int32)
            x_int = x_int.clamp(0, self.int_max - 1)

            if "BitLog2_Single" in self.log_quant_scheme:
                x_q, x_dq = self.bitLog2Single(x_int)

            elif "BitLog2_Half" in self.log_quant_scheme:
                x_q, x_dq = self.bitLog2Half(x_int)

            if self.inited is False:
                best_score, best_scale = 1e10, 1
                for i in torch.arange(x_dq.max(), self.int_max):
                    out = x_dq * 1 / i
                    score = lp_loss(x, out, p=2, reduction="all")

                    if score < best_score:
                        best_score, best_scale = score, i
                self.delta = best_scale
                print(f"self.delta: {self.delta}")

            x = x_dq * 1 / self.delta

            if self.inited is False:
                print(x_q.unique().numel(), x_q.unique())
                print(x_dq.unique().numel(), x_dq.unique())
                print(x.unique().numel(), x.unique())
                print()
                if self.int_max == 65536:
                    assert x.unique().numel() <= 17
                elif self.int_max == 32768:
                    assert x.unique().numel() <= 16
                elif self.int_max == 384:
                    assert x.unique().numel() <= 17
                elif self.int_max == 256:
                    assert x.unique().numel() <= 16
                self.inited = True
            return x

        elif "Sqrt2" in self.log_quant_scheme:
            """when using Original RepQ-ViT's Log(sqrt2) Code"""
            if self.inited is False:
                self.delta = self.init_quantization_scale(x)
                self.inited = True

            # start quantization
            x_dequant = self.quantize(x, self.delta)
            return x_dequant
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor):
        if x.min() < 0:
            # torch.save(x, "x.pt")
            x = x + 0.17
            x_max = x.max()
            x = x / x_max

            x = self.forward_logquant(x)
            x = x * x_max
            x = x - 0.17
            # torch.save(x, "x_q_dq.pt")
            # exit()
        else:
            x = self.forward_logquant(x)

        return x

    def init_quantization_scale(self, x: torch.Tensor):
        delta = None
        x_clone = x.clone().detach()
        delta = x_clone.max()
        best_score = 1e10
        for pct in [0.999, 0.9999, 0.99999]:  #
            try:
                new_delta = torch.quantile(x_clone.reshape(-1), pct)
            except:
                new_delta = torch.tensor(
                    np.percentile(x_clone.reshape(-1).cpu(), pct * 100),
                    device=x_clone.device,
                    dtype=torch.float32,
                )
            x_q = self.quantize(x_clone, new_delta)
            score = lp_loss(x_clone, x_q, p=2, reduction="all")
            if score < best_score:
                best_score = score
                delta = new_delta

        return delta

    def quantize(self, x, delta):
        from math import sqrt

        x_int = torch.round(-1 * (x / delta).log2() * 2)
        mask = x_int >= self.n_levels
        if self.log_quant_scheme == "Sqrt2_16":
            # Modified RepQ-ViT's CODE
            x_quant = torch.clamp(x_int, 0, self.n_levels - 2)
        elif self.log_quant_scheme == "Sqrt2_17":
            # ORIGINAL RepQ-ViT's CODE
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        else:
            raise NotImplementedError
        odd_mask = (x_quant % 2) * (sqrt(2) - 1) + 1
        x_float_q = 2 ** (-1 * torch.ceil(x_quant / 2)) * odd_mask * delta
        x_float_q[mask] = 0

        return x_float_q
