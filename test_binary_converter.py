#!/usr/bin/env python3
"""Tests for  binary_conversion_tools.py


This is a tool to turn pytorch's bfloat16 into binary tensors and back.
This code converts tensors of floats or bits into the respective other.
We use the google guideline [1] to convert. The default for conversion 
are based on bfloat 16 bit; 1 sign bit, 8 exponent bits and 7 mantissa bits.


num total bits     exponent bits   mantissa bits       bias
---------------------------------------------------------------
    16 bits               8                7           127

The correct answers for the test have been generated with help of this website:
https://www.h-schmidt.net/FloatConverter/IEEE754.html

[1] https://cloud.google.com/tpu/docs/bfloat16

Author, Karen Ullrich June 2019
Modified by Younghoon Byun, June 2023
"""
import numpy as np
import torch
from torch import nn


from binary_converter import bfloat162bit, bit2bfloat16

print("Testing the bfloat16 to bit conversion")

tensor = torch.rand(100)
tensor = tensor.type(torch.bfloat16)
target_bfloat16 = tensor

pred_bit = bfloat162bit(tensor, num_e_bits=8, num_m_bits=7, bias=127.)
pred_bfloat16 = bit2bfloat16(pred_bit)

assert len(tensor) == (target_bfloat16 == pred_bfloat16).sum(), "float2bit does not produce correct shape"

print("Test successful.")
