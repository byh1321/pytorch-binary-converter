# Binary Converter

This is a tool to turn pytorch's bfloat16 into binary tensors and back.
This code converts tensors of floats or bits into the respective other.
We use the google guideline [1] to convert. The default for conversion 
are based on bfloat 16 bit; 1 sign bit, 8 exponent bits and 7 mantissa bits.

|num total bits    | exponent bits |  mantissa bits   |    bias |
|------------|-------------------|-------------------|-------------------|  
|    16 bits  |        8       |      7     |        127|

### Usage

To turn a bfloat16 tensor into a binary one

    from binary_converter import bfloat162bit
    binary_tensor = bfloat162bit(float_tensor, num_e_bits=8, num_m_bits=7, bias=127.)

To turn a binary tensor into a float one

    from binary_converter import bit2bfloat16
    float_tensor = bit2bfloat16(binary_tensor, num_e_bits=8, num_m_bits=7, bias=127.)


### Requirements

This code has been tested with
-   `python 3.8`
-   `pytorch 1.13.0`

### Maintenance

Please be warned that this repository is not going to be maintained regularly.


### References

[1] https://cloud.google.com/tpu/docs/bfloat16
