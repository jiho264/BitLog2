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

    def int_log_quant_10x(self, x):
        """when using 4Bit INT Log2 Quantization"""
        x = x.to(torch.int32)
        zero_mask = x == 0
        log2_int = torch.full_like(x, -1, dtype=torch.int32)

        temp_x = x.clone()
        for i in range(15, -1, -1):
            shift = 1 << i
            greater_equal = temp_x >= shift
            log2_int += greater_equal.to(torch.int32)
            temp_x = temp_x >> greater_equal.to(torch.int32)

        fractional_add = torch.zeros_like(x, dtype=torch.int32)

        temp_x = x - (1 << log2_int)
        temp_x = temp_x << 1  # temp_x *= 2
        fractional_add += (temp_x >= (1 << log2_int)).to(torch.int32) * 5
        out = log2_int * 10 + fractional_add
        out[zero_mask] = -99999
        return out

    def int_log_dequant_10x(self, y):
        """when using 4Bit INT Log2 Quantization"""
        zero_mask = y < 0

        int_part = y // 10
        frac_part = y % 10 / 5

        int_num = 1 << int_part
        frac_num = frac_part * (1 << (int_part - 1))
        out = (int_num + frac_num).floor()
        out[zero_mask] = 0
        return out

    def forward(self, x: torch.Tensor):
        if "BitLog2_Single" in self.log_quant_scheme:
            """when using 4Bit INT Log2 Quantization"""
            if self.log_quant_scheme == "BitLog2_Single_16":
                int_max = 32768
            elif self.log_quant_scheme == "BitLog2_Single_17":
                int_max = 65536
            else:
                raise NotImplementedError

            x_int = torch.floor(x * int_max).to(torch.int32)
            x_int = x_int.clamp(0, int_max - 1)

            x_q = (self.int_log_quant_10x(x_int) // 10) * 10
            x_dq = self.int_log_dequant_10x(x_q)

            if self.inited is False:
                best_score = 1e10
                best_scale = 1
                for i in torch.arange(x_dq.max(), int_max):
                    out = x_dq * 1 / i
                    score = lp_loss(x, out, p=2, reduction="all")

                    if score < best_score:
                        best_score = score
                        best_scale = i
                self.delta = best_scale
                print(f"self.delta: {self.delta}")

            x = x_dq * 1 / self.delta

            if self.inited is False:
                print(x_q.unique().numel(), x_q.unique())
                print(x_dq.unique().numel(), x_dq.unique())
                print(x.unique().numel(), x.unique())
                if int_max == 65536:
                    assert x.unique().numel() <= 17
                elif int_max == 32768:
                    assert x.unique().numel() <= 16
                self.inited = True

            return x

        elif "BitLog2_Half" in self.log_quant_scheme:
            """when using 4Bit INT Log2 Half Quantization"""
            if self.log_quant_scheme == "BitLog2_Half_16":
                int_max = 256
            elif self.log_quant_scheme == "BitLog2_Half_17":
                int_max = 384
            else:
                raise NotImplementedError

            x_int = torch.floor(x * int_max).to(torch.int32)
            x_int = x_int.clamp(0, int_max - 1)

            x_q = self.int_log_quant_10x(x_int)
            x_dq = self.int_log_dequant_10x(x_q)

            if self.inited is False:
                best_score = 1e10
                best_scale = 1
                for i in torch.arange(x_dq.max(), int_max):
                    out = x_dq * 1 / i

                    score = lp_loss(x, out, p=2, reduction="all")
                    # print(f"scale: {i}, score: {score}")

                    if score < best_score:
                        best_score = score
                        best_scale = i
                self.delta = best_scale
                print(f"self.delta: {self.delta}")

            x = x_dq * 1 / self.delta

            if self.inited is False:
                print(x_q.unique().numel(), x_q.unique())
                print(x_dq.unique().numel(), x_dq.unique())
                print(x.unique().numel(), x.unique())
                print()
                if int_max == 384:
                    assert x.unique().numel() <= 17
                elif int_max == 256:
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
