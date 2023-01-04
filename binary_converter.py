#!/usr/bin/env python3
"""Converting between floats and binaries.

This code converts tensors of floats or bits into the respective other.
We use the google guideline [1] to convert. The default for conversion 
are based on bfloat 16 bit; 1 sign bit, 8 exponent bits and 7 mantissa bits.
Other common formats are

num total bits     exponent bits   mantissa bits       bias
---------------------------------------------------------------
    16 bits               8                7           127

Available modules:
    * bit2bfloat16
    * bfloat162bit
    * integer2bit
    * remainder2bit

[1] https://cloud.google.com/tpu/docs/bfloat16

Author, Karen Ullrich June 2019
Modified by Younghoon Byun, June 2023
"""

import torch
import warnings


def bit2bfloat16(b, num_e_bits=8, num_m_bits=7, bias=127.):
  """Turn input tensor into float.
      Args:
          b : binary tensor. The last dimension of this tensor should be the
          the one the binary is at.
          num_e_bits : Number of exponent bits. Default: 8.
          num_m_bits : Number of mantissa bits. Default: 7.
          bias : Exponent bias/ zero offset. Default: 127.
      Returns:
          Tensor: Float tensor. Reduces last dimension.
  """
  expected_last_dim = num_m_bits + num_e_bits + 1
  assert b.shape[-1] == expected_last_dim, "Binary tensors last dimension " \
                                           "should be {}, not {}.".format(
    expected_last_dim, b.shape[-1])

  # check if we got the right type
  dtype = torch.Tensor
  if expected_last_dim > 32: dtype = torch.float64
  if expected_last_dim > 64:
    warnings.warn("pytorch can not process floats larger than 64 bits, keep"
                  " this in mind. Your result will be not exact.")

  s = torch.index_select(b, -1, torch.arange(0, 1))
  e = torch.index_select(b, -1, torch.arange(1, 1 + num_e_bits))
  m = torch.index_select(b, -1, torch.arange(1 + num_e_bits,
                                             1 + num_e_bits + num_m_bits))
  # SIGN BIT
  out = ((-1) ** s).squeeze(-1).type(dtype)
  # EXPONENT BIT
  exponents = -torch.arange(-(num_e_bits - 1.), 1.)
  exponents = exponents.repeat(b.shape[:-1] + (1,))
  e_decimal = torch.sum(e * 2 ** exponents, dim=-1) - bias
  subnormal_correction_tensor = (e_decimal == -127).type(dtype)
  e_decimal = torch.where(e_decimal == -127, -126, e_decimal)
  out *= 2 ** e_decimal
  # MANTISSA
  matissa = (torch.Tensor([2.]) ** (
    -torch.arange(1., num_m_bits + 1.))).repeat(
    m.shape[:-1] + (1,))
  out *= torch.ones(b.shape[0]) - subnormal_correction_tensor + torch.sum(m * matissa, dim=-1)
  return out


def bfloat162bit(f, num_e_bits=8, num_m_bits=7, bias=127.):
  """Turn input tensor into binary.
      Args:
          f : bfloat16 tensor.
          num_e_bits : Number of exponent bits. Default: 8.
          num_m_bits : Number of mantissa bits. Default: 7.
          bias : Exponent bias/ zero offset. Default: 127.
      Returns:
          Tensor: Binary tensor. Adds last dimension to original tensor for
          bits.
  """
  ## SIGN BIT
  s = torch.sign(f)
  f = f * s
  # turn sign into sign-bit
  s = (s * (-1) + 1.) * 0.5
  s = s.unsqueeze(-1)

  ## EXPONENT BIT
  e_scientific = torch.floor(torch.log2(f))
  e_decimal = e_scientific + bias
  e = integer2bit(e_decimal, num_bits=num_e_bits)

  ## MANTISSA
  m1 = integer2bit(f - f % 1, num_bits=num_e_bits)
  m2 = remainder2bit(f % 1, num_bits=bias)
  m = torch.cat([m1, m2], dim=-1)
  
  dtype = f.type()
  idx = torch.arange(num_m_bits).unsqueeze(0).type(dtype) \
        + (8. - e_scientific).unsqueeze(-1)
  idx = idx.long()
  idx = torch.where(idx < -1, 0, idx)
  m = torch.gather(m, dim=-1, index=idx)
  out = torch.cat([s, e, m], dim=-1).type(dtype)
  zero_bin = torch.zeros(16)
  for i in range(out.shape[0]):
      if out[i,0] == 0.5:
          out[i] = zero_bin
  return out


def remainder2bit(remainder, num_bits=127):
  """Turn a tensor with remainders (floats < 1) to mantissa bits.
      Args:
          remainder : torch.Tensor, tensor with remainders
          num_bits : Number of bits to specify the precision. Default: 127.
      Returns:
          Tensor: Binary tensor. Adds last dimension to original tensor for
          bits.
  """
  dtype = remainder.type()
  exponent_bits = torch.arange(num_bits).type(dtype)
  exponent_bits = exponent_bits.repeat(remainder.shape + (1,))
  out = (remainder.unsqueeze(-1) * 2 ** exponent_bits) % 1
  return torch.floor(2 * out)


def integer2bit(integer, num_bits=8):
  """Turn integer tensor to binary representation.
      Args:
          integer : torch.Tensor, tensor with integers
          num_bits : Number of bits to specify the precision. Default: 8.
      Returns:
          Tensor: Binary tensor. Adds last dimension to original tensor for
          bits.
  """
  dtype = integer.type()
  exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
  exponent_bits = exponent_bits.repeat(integer.shape + (1,))
  out = integer.unsqueeze(-1) / 2 ** exponent_bits
  return (out - (out % 1)) % 2
