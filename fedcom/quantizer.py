import numpy as np

# PyTorch Libraries
import torch

from fedcom import Quantizer

class UniformQuantizer(Quantizer):
    def __init__(self, config):
        self.quantbound = config.quantization_level - 1     
        self.debug_mode = config.debug_mode

    def quantize(self, arr):
        """
        quantize a given arr array with unifrom quantization.
        """
        max_val = torch.max(arr.abs())
        sign_arr = arr.sign()
        quantized_arr = (arr/max_val)*self.quantbound
        quantized_arr = torch.abs(quantized_arr)
        quantized_arr = torch.round(quantized_arr).to(torch.int)
        
        quantized_set = dict(max_val=max_val, signs=sign_arr, quantized_arr=quantized_arr)
        return quantized_set
    
    def dequantize(self, quantized_set):
        """
        dequantize a given array which is uniformed quantized.
        """
        coefficients = quantized_set["max_val"]/self.quantbound  * quantized_set["signs"] 
        dequant_arr =  coefficients * quantized_set["quantized_arr"]

        return dequant_arr

class QsgdQuantizer(Quantizer):

    def __init__(self, config):
        self.quantlevel = config.quantization_level 
        self.quantbound = config.quantization_level - 1
        self.debug_mode = config.debug_mode

    def quantize(self, arr):
        norm = arr.norm()
        abs_arr = arr.abs()

        level_float = abs_arr / norm * self.quantbound 
        lower_level = level_float.floor()
        rand_variable = torch.empty_like(arr).uniform_() 
        is_upper_level = rand_variable < (level_float - lower_level)
        new_level = (lower_level + is_upper_level)
        quantized_arr = torch.round(new_level).to(torch.int)

        sign = arr.sign()
        quantized_set = dict(norm=norm, signs=sign, quantized_arr=quantized_arr)

        if self.debug_mode:
            quantized_set["original_arr"] = arr.clone()

        return quantized_set

    def dequantize(self, quantized_set):
        coefficients = quantized_set["norm"]/self.quantbound * quantized_set["signs"]
        dequant_arr = coefficients * quantized_set["quantized_arr"]

        return dequant_arr


# A plain quantizer that does nothing. (for vanilla FedAvg)
class PlainQuantizer(Quantizer):
    def __init__(self, config):
        self.debug_mode = config.debug_mode

    def quantize(self, arr):
        """
        simply return the arr.
        """
        quantized_set = dict(quantized_arr=arr)
        if self.debug_mode:
            quantized_set["original_arr"] = quantized_set["quantized_arr"]
        return quantized_set
    
    def dequantize(self, quantized_set):
        """
        dequantize a given array which is uniformed quantized.
        """
        dequant_arr = quantized_set["quantized_arr"]

        return dequant_arr