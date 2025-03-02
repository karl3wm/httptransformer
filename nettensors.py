# NetTensor uses the internal rather than public interface of NetSafeTensors
# this is to provide arbitrary tensor slicing but take advantage of memory mapped caching
import netsafetensors

import math

import psutil, torch, tqdm

__nettensor_torch_impls__={}
__nettensor_torch_fetch__=set()
# choice of subclassing rather than composing Tensor so that uses of isinstance() succeed
class NetTensor(torch.Tensor):
    # NetTensor is a 'meta' tensor but fakes its device
    device = torch.device('cpu')

    # the code might simplify if the constructor signature were made to match the assumptions of torch's _convert function
    def __new__(cls, safeslice, tensor = None):
        if tensor is None:
            tensor = safeslice.tensor
        self = super().__new__(cls, tensor)
        self.safeslice = safeslice
        return self

    def __getitem__(self, slice):
        subtensor = super().__getitem__(slice)
        return type(self)(self.safeslice, subtensor)

    def mem_usage_frac(self):
        bytes_avail_cpu = psutil.virtual_memory().available * self.safeslice.statedict.usage_frac
        return self.numel() * self.dtype.itemsize / bytes_avail_cpu

    def fetch(self, progress=''):
        assert self.mem_usage_frac() < 1
        count = self.numel()
        readsize = self.safeslice.tensor.element_size()
        readlength = count * readsize
        offset = self.safeslice.offset + self.storage_offset() * readsize
        data = self.safeslice.fetcher.read(offset, readlength, progress=progress)
        return torch.frombuffer(
            data,
            dtype=self.safeslice.tensor.dtype,
            count=count,
        ).view(self.shape).to(self.device, self.dtype)

    def clone(self):
        return type(self)(self.safeslice, self)

    def to(self, dtype_or_device, dtype_or_nonblocking = None, nonblocking = None):
        if type(dtype_or_device) is torch.dtype:
            dtype = dtype_or_device
            if dtype != self.dtype:
                self = type(self)(self.safeslice, super().to(dtype))
        else:
            device = torch.device(dtype_or_device)
            if type(dtype_or_nonblocking) is torch.dtype:
                dtype = dtype_or_nonblocking
                if dtype != self.dtype or device != self.device:
                    self = type(self)(self.safeslice, super().to(dtype))
                    self.device = device
            else:
                if device.type not in [self.device.type, 'meta']:
                    self = self.clone()
                    self.device = device
        return self

    def contiguous(self):
        return self

    @staticmethod
    def __provides(torch_func):
        return lambda func: __nettensor_torch_impls__.__setitem__(torch_func, func)
    #def __default(torch_func):
    #    __nettensor_torch_dflts__.add(torch_func)
    def __fetch(torch_func):
        __nettensor_torch_fetch__.add(torch_func)

    __fetch(torch.Tensor.mul)

    @__provides(torch.nn.functional.embedding)
    def F_embedding(input, weight, *params, **kwparams):
        tokens, dense_input = input.unique(sorted = False, return_inverse = True)
        dense_embedding = torch.stack([
            weight[token].fetch()
            for token in tokens
        ])
        return torch.nn.functional.embedding(dense_input, dense_embedding, *params, **kwparams)

    @__provides(torch.nn.functional.linear)
    def F_linear(input, weight, bias=None):
        cls = NetTensor
        assert type(weight) is cls and type(input) is torch.Tensor
        assert bias is None
        name = weight.safeslice.name.rsplit('.',1)[0]
        number_passes = math.ceil(weight.mem_usage_frac())
        if number_passes == 1:
            return torch.matmul(input, weight.fetch(progress=name).T)
        else:
            rows_at_once = math.ceil(weight.shape[0] / number_passes)
            return torch.cat([
                torch.matmul(
                    input,
                    weight[offset : offset+rows_at_once].fetch(progress=f'row{offset}-{offset+rows_at_once}/{weight.shape[0]}').T
                )
                for offset in tqdm.tqdm(range(
                    0,
                    weight.shape[0],
                    rows_at_once
                ), desc=name, unit='blk', leave=False)
            ], dim=-1)

    @classmethod
    def __torch_function__(cls, func, types, params=[], kwparams={}):
        handle = __nettensor_torch_impls__.get(func)
        if handle is None:
            if func in __nettensor_torch_fetch__:
                if type(params[0]) is cls:
                    params = [params[0].fetch(), *params[1:]]
                else:
                    params = [params[0], params[1].fetch(), *params[2:]]
                result = torch.Tensor.__torch_function__(func, [], params, kwparams)
            else:
                result = torch.Tensor.__torch_function__(func, [], params, kwparams)
                if type(result) is torch.Tensor:
                    result = cls(params[0].safeslice, result)
            return result
        else:
            return handle(*params, **kwparams)

class NetStateDict:
    def __init__(self, safetensors, usage_frac = 0.5):
        self.safetensors = safetensors
        self.keys = self.safetensors.keys
        self.usage_frac = usage_frac
    def keys(self):
        return self.safetensors.keys()
    def __getitem__(self, item):
        safeslice = self.safetensors.get_slice(item)
        safeslice.statedict = self
        safeslice.name = item
        return NetTensor(safeslice)
    def __iter__(self):
        return iter(self.keys())
    def values(self):
        return (self[k] for k in self.safetensors.keys())
    def items(self):
        return ([k, self[k]] for k in self.safetensors.keys())

def from_hf_hub(repo_id, lfs_filename = None, revision='main', repo_type=None, **kwparams):
    return NetStateDict(netsafetensors.from_hf_hub(repo_id, lfs_filename, revision, repo_type, **kwparams))

