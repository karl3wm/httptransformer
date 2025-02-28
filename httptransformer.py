import contextlib, inspect, json, os, psutil
import accelerate, huggingface_hub, requests, torch, tqdm, transformers

class Quirks:
    @contextlib.contextmanager
    def fake_cuda_available():
        if not torch.cuda.is_available():
            cached_is_available = torch.cuda.is_available
            cached_get_device_capability = torch.cuda.get_device_capability
            torch.cuda.is_available = lambda: True
            torch.cuda.get_device_capability = lambda: [9,0]
            try:
                yield
            finally:
                torch.cuda.is_available = cached_is_available
                torch.cuda.get_device_capability = cached_get_device_capability
        else:
            yield
    #def unify_rope(model_or_class):
    #    # deepseek generates separate rope buffers for each layer which can use significant memory
    #    deepseek = inspect.getmodule(model_or_class)
    #    cache = {}
    #    def wrap_rope(rope):
    #        def wrapper(*params, **kwparams):
    #            key = tuple(params) + tuple(kwparams)
    #            if key in cache:
    #                return cache[key]
    #            else:
    #                val = rope(*params, **kwparams)
    #                cache[key] = val
    #                return val
    #        return wrapper
    #    for key, val in deepseek.__dict__.items():
    #        if 'Rotary' in key and isinstance(val, torch.nn.Module):
    #            setattr(deepseek, key, wrap_rope(val))

LAZY_TENSOR_TORCH_FUNCTIONS = {}
class LazyTensor(torch.Tensor):

    session = requests.Session()

    @classmethod
    def from_json(cls, url, N, weight, data, device):
        start, end = data['data_offsets']
        start += N + 8
        end += N + 8 - 1
        result = cls.decode([
            weight, url, start, data['shape'], data['dtype'], device
        ])
        assert end - start + 1 == result.nbytes
        result = cls.decode(result.encode())
        return result

    def encode(self):
        dtype = str(self.dtype).removeprefix('torch.')
        dtype = dtype.replace('float', 'F')
        dtype = dtype.removesuffix('fn')
        dtype = dtype.upper()
        return [self.name_, self.request.url, self.storage_offset(), list(self.shape), dtype, self.target_device.type]
    @classmethod
    def decode(cls, data):
        weight, url, storage_offset, shape, dtype, device = data
        dtype = dtype.replace('F', 'float')
        dtype = dtype.lower()
        try:
            dtype = getattr(torch, dtype)
        except:
            dtype = getattr(torch, dtype + 'fn')
        tensor = cls(torch.empty(shape, dtype = dtype, device = 'meta'))
        tensor.set_(source = tensor.untyped_storage(), storage_offset = storage_offset, size = tensor.size())
        start = storage_offset
        end = storage_offset + tensor.nbytes - 1
        request = requests.Request('GET', url, dict(Range='bytes='+str(start)+'-'+str(end)))
        request = request.prepare()
        tensor.name_ = weight
        tensor.request = request
        tensor.target_device = torch.device(device)
        return tensor

    def _set(self, other):
        self.name_ = other.name_
        self.request = other.request
        self.target_device = other.target_device
        return self

    def _adjust(self, other):
        self.name_ = other.name_
        self.target_device = other.target_device
        assert self.is_contiguous()
        start = self.storage_offset()
        size = self.nbytes
        #for stride in self.stride():
        #    size *= stride
        end = start + size - 1
        self.request = other.request.copy()
        self.request.prepare_headers(dict(Range='bytes='+str(start)+'-'+str(end)))
        return self

    def fits_in_memory(self):
        return self.nbytes < psutil.virtual_memory().available / 2

    def download(self):
        chunk_size = 1024*128
        with tqdm.tqdm(desc=self.name_, leave=False, total=self.nbytes) as pbar:
            buffer = memoryview(bytearray(self.nbytes))

            with self.session.send(self.request, stream=True) as response:
                while pbar.n < pbar.total:
                    pbar.update(
                        response.raw.readinto(
                            buffer[ pbar.n : pbar.n + chunk_size ]
                        )
                    )
        return torch.frombuffer(
            buffer,
            dtype = self.dtype,
            count = self.numel(),
            requires_grad = False,
        ).to(self.target_device).reshape(self.shape)


    # torch api implementations

    @classmethod
    def __torch_function__(cls, func, types, params=[], kwparams={}):
        handle = LAZY_TENSOR_TORCH_FUNCTIONS.get(func)
        if handle is None:
            result = super().__torch_function__(func, types, params, kwparams)
            if type(result) is cls and func not in [torch.Tensor.detach,torch.Tensor.to,torch.Tensor.__getitem__]:
                if not hasattr(result, 'target_device'):
                    raise NotImplementedError(func)
            return result
        else:
            return handle(*params, **kwparams)
    @staticmethod
    def __provides(torch_func):
        return lambda func: LAZY_TENSOR_TORCH_FUNCTIONS.__setitem__(torch_func, func)

    def clone(self):
        return super().clone()._set(self)
    def detach(self):
        return super().detach()._set(self)
    def to(self, device_or_dtype):
        assert device_or_dtype in [self.device, self.device.type, self.target_device, self.target_device.type, self.dtype] # stub
        return self
    def __getitem__(self, idcs):
        return super().__getitem__(idcs)._adjust(self)

    @__provides(torch.nn.functional.embedding)
    def __embedding(input, weight, *params, **kwparams):
        tokens, dense_input = input.unique(sorted = False, return_inverse = True)
        dense_embedding = torch.stack([
            weight[token].download()
            for token in tokens
        ])
        return torch.nn.functional.embedding(dense_input, dense_embedding, *params, **kwparams)



class LazyStateDict(dict):
    def __init__(self, tensor_request_by_name, device):
        super().__init__(tensor_request_by_name)
        self.device = device

    def get_meta_tensor(self, weight):
        return super().__getitem__(weight)[0]

    def __getitem__(self, weight):
        tensor = super().__getitem__(weight)
        if tensor.fits_in_memory():
            return tensor.download()
        else:
            return tensor

    def items(self):
        for key in self:
            yield [key, self[key]]

    def values(self):
        for key in self:
            yield self[key]

    def largest(self):
        return max([
            [key, tensor]
            for key, [tensor, _] in super().items()
        ], key = lambda keytensor: keytensor[1].nbytes)[1]

    @classmethod
    def from_user_repo_branch(cls, user, repo, branch, device):
        base_url = f'https://huggingface.co/{user}/{repo}/raw/{branch}/'
        lfs_base_url = f'https://huggingface.co/{user}/{repo}/resolve/{branch}/'

        safetensors_index_fn = huggingface_hub.hf_hub_download(user + '/' + repo, 'model.safetensors.index.json', revision=branch)

        tensors = {}

        #cache_fn = os.path.join(os.path.dirname(safetensors_index_fn), 'httptransformer.json')
        cache_fn = safetensors_index_fn + '.httptransformer.json'
        if os.path.exists(cache_fn):
            with open(cache_fn) as cache_fh, tqdm.tqdm(desc='cached tensor urls', unit='B', unit_scale=True) as pbar:
                pbar.total = cache_fh.seek(0, os.SEEK_END)
                cache_fh.seek(0, os.SEEK_SET)
                while True:
                    line = cache_fh.readline()
                    if not line:
                        break
                    tensor = LazyTensor.decode(json.loads(line))
                    tensors[tensor.name_] = tensor
                    pbar.update(cache_fh.tell() - pbar.n)
        if len(tensors) == 0:
            #safetensors_index_url = base_url + 'model.safetensors.index.json'
            #print(safetensors_index_url)
            #with requests.get(safetensors_index_url, stream=True) as response:
            #    safetensors_index = json.load(response.raw)
            with open(safetensors_index_fn) as safetensors_index_fh:
                safetensors_index = json.load(safetensors_index_fh)
    
            print(safetensors_index['metadata'])
    
            fn_by_weight = safetensors_index['weight_map']
    
            urls = [lfs_base_url + fn for fn in set(fn_by_weight.values())]

            tensors = {}
        
            with open(cache_fn, 'w') as cache_fh, tqdm.tqdm(urls,desc='downloading tensor urls (rm cache if interrupted!)') as pbar:
              for url in pbar:
                # we could potentially also check the git-lfs sha256 from the base url and merklify the data too, this would mean downloading it all
                #[b'version https://git-lfs.github.com/spec/v1', b'oid sha256:e94d32e8649e1a5b03cc0a343c59ca5a6d80d03cd46161b482fd3bb2484adb7d', b'size 4302350824']
                #lfs = dict([ line.decode().split(' ', 1) for line in response.iter_lines() ])
                with requests.get(url, stream=True) as response:
                    N = int.from_bytes(response.raw.read(8), 'little')
                    header = json.loads(response.raw.read(N))
                for weight, data in header.items():
                    if weight == '__metadata__':
                        continue
                    #tensor_request_by_name[weight] = cls.tensor_request_from_json(url, N, data)
                    tensor = LazyTensor.from_json(url, N, weight, data, device)
                    tensors[weight] = tensor
                    json.dump(tensor.encode(), cache_fh)
                    cache_fh.write('\n')

        return cls(tensors, device)

def construct(model_id, device, config_patches = {}, attr_patches = {}):
    user, repo = model_id.split('/',1)
    branch = 'main'

    print(user, repo, branch)

    config = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    for key, val in config_patches.items():
        setattr(config, key, val)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    with accelerate.init_empty_weights(), transformers.modeling_utils.no_init_weights():
        model = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    for key, val in attr_patches.items():
        setattr(model, key, val)

    lazy_state_dict = LazyStateDict.from_user_repo_branch(user, repo, branch, device=device)

    # misuse cpu offloading by providing lazy_state_dict
    model = accelerate.cpu_offload(model, device, state_dict = lazy_state_dict)
    model.hf_device_map = { '': device }

    model.eval()
    return transformers.pipeline('text-generation', model=model, config=config, tokenizer=tokenizer)

with Quirks.fake_cuda_available(): # avoid FP8 assertions on cpu during placement testing
    pipe = construct(
        'deepseek-ai/DeepSeek-V3',
        device = 'cpu',
        config_patches = dict(
            max_position_embeddings = 64, # drop ctx len from 163840 to 64
        ),
        attr_patches = dict(
            _supports_cache_class = False, # might be a bug that this isn't in the model
        ),
    )
pipe('Once upon a time,', streamer = transformers.TextStreamer(pipe.tokenizer))
