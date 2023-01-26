import torch
import torch.nn as nn
import pdb
#import matplotlib.pyplot as plt
#import seaborn as sns
import math
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class ILSLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.nn.Parameter, cache, compute, Qflag):
        if cache:
            if Qflag:
                ctx.shape = input.shape
                ctx.save_for_backward(shaper(quantize(input, 3, 4), '8-bit'), weight)
            else:
                ctx.save_for_backward(input, weight)
        elif compute:
            ctx.save_for_backward(None, weight)
        ctx.cache = cache
        ctx.compute = compute
        ctx.Qflag = Qflag
        return nn.functional.linear(input, weight)


    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.cache:
            if ctx.Qflag:
                x, weight = deshaper(ctx.saved_tensors[0],'8-bit').reshape(ctx.shape)/2**4, ctx.saved_tensors[1]
                grad_weight = grad_output.reshape(-1,grad_output.shape[-1]).T.mm(x.reshape(-1,x.shape[-1]))
                grad_input = torch.matmul(grad_output, weight)
            else:
                x, weight = ctx.saved_tensors
                grad_weight = grad_output.reshape(-1,grad_output.shape[-1]).T.mm(x.reshape(-1,x.shape[-1]))
                grad_input = torch.matmul(grad_output, weight)
        elif ctx.compute:
            x, weight = ctx.saved_tensors
            grad_input = torch.matmul(grad_output, weight)
            grad_weight = None
        else:
            grad_input = None
            grad_weight = None
        return grad_input, grad_weight, None, None, None



class ILSLinear(nn.Linear):
    def __init__(self, *kargs, bias=True, Qflag = False):
        super(ILSLinear, self).__init__(*kargs, bias=True)
        self.Qflag = Qflag
    def forward(self, input, cache = True, compute = True):
        if self.weight.requires_grad:
            cache = True
        else:
            cache = False
        out = ILSLinearFunction.apply(input, self.weight, cache, compute, self.Qflag)
        #out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out) 

        return out

####################################################
#Note: We used select_mink and pad_mink from https://github.com/chenjoya/dropit
####################################################
def select_mink(reserve, x: torch.Tensor, ctx=None):
    ctx.x_shape = x.shape
    
    numel = x.numel()
    x = x.reshape(-1)
    idxs = x.abs().topk(int(numel * reserve), sorted=False)[1]
    x = x[idxs]
    ctx.idxs = idxs.to(torch.int32)
    ctx.numel = numel
    return x

def pad_mink(x, ctx=None):      
    idxs = ctx.idxs.to(torch.int64)
    del ctx.idxs
    return torch.zeros(
        ctx.numel, device=x.device, dtype=x.dtype
    ).scatter_(0, idxs, x).reshape(ctx.x_shape)

def quantize(input: torch.Tensor, int_bits, frac_bits):
    
    return torch.clamp(torch.round(input * 2**frac_bits), -2**(int_bits+frac_bits), 2**(int_bits+frac_bits) - 1)

def shaper(input: torch.Tensor, type):
    if type == '4-bit':
        input = torch.tensor_split(input.reshape(-1), 2)
        input = (input[0] + 8) * 2**4 + (input[1] + 8)
        input = input.type(torch.cuda.ByteTensor)
    elif type == '8-bit':
        input = input.type(torch.cuda.CharTensor)
    else:
        input

    return input


def deshaper(input: torch.Tensor, type):
    if type == '4-bit':
        input = torch.cat( (torch.div(input, 2**4,rounding_mode="floor"), torch.remainder(input, 2**4)), 0) 
        input = input.type(torch.cuda.FloatTensor) - 8.
    elif type == '8-bit':
        input = input.type(torch.cuda.FloatTensor)
    else:
        input

    return input


class ILSLayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, normalized_shape, weight: torch.nn.Parameter, bias: torch.nn.Parameter, eps, cache, compute):
        ctx.cache = cache
        ctx.compute = compute
        if cache:
            ctx.save_for_backward(input, weight)
            #ctx.save_for_backward(shaper(quantize(input, 5, 2), '8-bit'), weight)
        elif compute:
            mean = input.mean(dim = -1, keepdim=True)
            var = ((input - mean) ** 2).mean(dim = -1, keepdim=True)
            std = (var + 1e-12).sqrt()
            y = (input - mean) / std
            ctx.save_for_backward(select_mink(0.1, y, ctx), weight, std)

        return nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)


    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.cache:
            x, weight = ctx.saved_tensors
            #x = x/2**2
            mean = x.mean(dim = -1, keepdim=True)
            var = ((x - mean) ** 2).mean(dim = -1, keepdim=True)
            std = (var + 1e-12).sqrt()
            y = (x - mean) / std

            N = grad_output.shape[-1]
            grad_weight = (grad_output * y).sum(dim = [0, 1])
            grad_bias = grad_output.sum(dim = [0, 1])
            dxhat = grad_output * weight * (1. / N) * (1/std)

            grad_input = (N*dxhat - dxhat.sum(dim = -1, keepdim=True) -  (dxhat*y).sum(dim = -1, keepdim=True) * y)



            #grad_input = (1 / N) * weight * (1/std * (N * grad_output.T) - grad_output.T.sum(dim = [int(s) for s in range(2,grad_output.dim())], keepdim=True) - (y.T * torch.square(1/std) * (grad_output.T*y).sum(dim = [int(s) for s in range(2,grad_output.dim())], keepdim=True) ))

        elif ctx.compute:
            x, weight, std = ctx.saved_tensors
            y = pad_mink(x, ctx)
            N = grad_output.shape[-1]
            grad_weight = None 
            grad_bias = grad_output.sum(dim = [0, 1])
            dxhat = grad_output * weight * (1. / N) * (1/std)

            grad_input = (N*dxhat- dxhat.sum(dim = -1, keepdim=True)  - (dxhat * y).sum(dim = -1, keepdim=True) * y)
        else:
            grad_input = None
            grad_weight = None
            grad_bias = None

        return grad_input, None, grad_weight, grad_bias, None, None, None


class ILSLayerNorm(nn.LayerNorm):
    def __init__(self, *kargs, eps):
        super(ILSLayerNorm, self).__init__(*kargs, eps)

    def forward(self, input, cache = True, compute = True):
        if self.weight.requires_grad:
            cache = True
        else:
            cache = False
        #out = nn.functional.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps) #
        out = ILSLayerNormFunction.apply(input, self.normalized_shape, self.weight, self.bias, self.eps, cache, compute)


        return out


class ILSConv2d(nn.Conv2d):
    # To be completed
    def __init__(self,  *kargs, bias=True, config=None):
        super(ILSConv2d, self).__init__(*kargs,bias=True)

 


    def forward(self, input, type=None):
        
        out = nn.functional.conv2d(input, self.weight, stride=self.config.patch_size) #, bias = self.bias
        
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out





class ILSEmbedding(nn.Embedding):
    # To be completed
    def __init__(self, *kargs, padding_idx = None):
        super(ILSEmbedding, self).__init__(*kargs, padding_idx = padding_idx)
        

    def forward(self, input, type=None):
        
        out = nn.functional.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        return out





class ILSGELUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor):
        #print(( (input<-0.) * (input>-3.0)  ).sum()/input.reshape(-1).shape[0])  select_mink(1., input, ctx)

        #ctx.save_for_backward( torch.clamp(torch.round(input * 2**1) / 2**1, -4, 3.5) )
        ctx.shape = input.shape
        ctx.save_for_backward( shaper(quantize(input, 2, 1), '4-bit') )
        #ctx.save_for_backward( input )
        return nn.functional.gelu(input)


    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        
        x = deshaper(ctx.saved_tensors[0],'4-bit').reshape(ctx.shape) / 2**1
        y = x #pad_mink(x, ctx)
        cdf = 0.5 * (1.0 + torch.erf(y / np.sqrt(2.0)))
        p = torch.exp(-0.5*y**2)/np.sqrt(2.*np.pi)
        grad_input = cdf + y * p
        
        return grad_input * grad_output


ILSGELU = ILSGELUFunction.apply

'''class ILSGELU(nn.Module):
    # To be completed
    def __init__(self):
        super().__init__()



    def forward(self, input):
        
        out = ILSGELUFunction.apply(input)

        return out'''



class ILSmatmulFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input1: torch.Tensor, input2: torch.Tensor):
        ctx.shape1 = input1.shape
        ctx.shape2 = input2.shape
        #print(input1.max(), input1.min(), input1.std(), input1.var(), input2.max(), input2.min(), input2.std(), input2.var())
        ctx.save_for_backward(shaper(quantize(input1, 3, 4), '8-bit'), shaper(quantize(input2, 3, 4), '8-bit'))
        #ctx.save_for_backward(input1,input2)
        return torch.matmul(input1, input2)


    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        in1, in2 = deshaper(ctx.saved_tensors[0],'8-bit').reshape(ctx.shape1)/2**4, deshaper(ctx.saved_tensors[1],'8-bit').reshape(ctx.shape2)/2**4
        #in1, in2 = ctx.saved_tensors[0], ctx.saved_tensors[1]
        #print(in1.min(),in1.max(), in1.var(),in2.min(),in2.max(), in2.var())
        grad_input2 = torch.matmul(in1.transpose(-1, -2), grad_output)
        grad_input1 = torch.matmul(grad_output, in2.transpose(-1, -2))
        return grad_input1, grad_input2 


ILSmatmul = ILSmatmulFunction.apply





class ILSDropout(nn.Dropout):
    def __init__(self, *kargs):
        super(ILSDropout, self).__init__(*kargs)
        self.binomial = torch.distributions.binomial.Binomial(probs=1-torch.as_tensor(self.p))
    def forward(self, input):
        return nn.functional.dropout(input, self.p, self.training, self.inplace)



