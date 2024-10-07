## BitLog2: A Log-Based 4-Bit Quantization for Attention Map Using Bit Shifting
![img](BitLog2.png)

- The 4-Bit Log-based Quantization mathod for attention map in Vision Transformer.
- Previously, RepQ-ViT employed a log base of √2 for quantizing the attention map.
- The exact value of the integer part of log2(x) can be calculated using a bit shift operation, avoiding the need for floating-point operations.
- **We propose the BitLog2 Half which separates the decimal part using bit shift operations. As a result, this achieves better accuracy than RepQ-ViT with using only Bit Shift operations.**
- Our experiments are based on RepQ-ViT, CVPR2023, but we replaced only the quantization method for the attention map.

Below are the instructions for reproducing the classification results of BitLog2.

## Evaluation

- You can quantize and evaluate a single model using the following command:

```bash
python test_quant.py [--model] [--dataset] [--w_bit] [--a_bit]

optional arguments:
--model: Model architecture, the choises can be: 
    vit_small, vit_base, deit_tiny, deit_small, deit_base, swin_tiny, swin_small.
--dataset: Path to ImageNet dataset.
--w_bit: Bit-precision of weights, default=4.
--a_bit: Bit-precision of activation, default=4.
--log_quant_scheme: Log Quantization Method, default=Sqrt2_17 from RepQ-ViT.
    Sqrt2_17, BitLog2_Single_17, BitLog2_Half_16, BitLog2_Half_17
```

- Example: Quantize *DeiT-S* at W4/A4 precision:

```bash
python test_quant.py --model deit_small --dataset <YOUR_DATA_DIR> --log_quant_scheme BitLog2_Half_17
```

## Results

Below are the experimental results of our proposed BitLog2 that you should get on ImageNet dataset.

| Model  | FP32  | RepQ-ViT | Single_17 |  Half_17   | Half_16 |
| :----: | :---: | :------: | :-------: | :--------: | :-----: |
| ViT-S  | 81.39 |  65.05   |  64.456   | **65.874** | 64.580  |
| ViT-B  | 84.54 |  68.48   |  66.824   | **68.900** | 67.482  |
| DeiT-T | 72.21 |  57.43   |  57.096   | **58.346** | 57.664  |
| DeiT-S | 79.85 |  69.03   |  68.716   | **69.554** | 69.432  |
| DeiT-B | 81.80 |  75.61   |  75.482   | **75.836** | 75.554  |


## Acknowledgement

This experiment is based on the [RepQ-ViT, CVPR2023](https://github.com/zkkli/RepQ-ViT).
Thanks to the author (zkkli) for providing the codebase.

<!-- | Swin-T (81.35) | W4/A4 |  72.31   | W6/A6 |  80.69   | -->
<!-- | Swin-S (83.23) | W4/A4 |  79.45   | W6/A6 |  82.79   | -->

<!-- ## Citation

We appreciate it if you would please cite the following paper if you found the implementation useful for your work:

```bash
@inproceedings{li2023repq,
  title={Repq-vit: Scale reparameterization for post-training quantization of vision transformers},
  author={Li, Zhikai and Xiao, Junrui and Yang, Lianwei and Gu, Qingyi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={17227--17236},
  year={2023}
}
``` -->
