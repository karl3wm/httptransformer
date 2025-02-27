import inspect, json, psutil
import accelerate, requests, torch, tqdm, transformers

class Quirks:
    if not torch.cuda.is_available():
        # this avoids FP8 assertions on cpu during placement testing
        torch.cuda.is_available = lambda: True
        torch.cuda.get_device_capability = lambda: [9,0]
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

class LazyStateDict(dict):
    def __init__(self, tensor_request_by_name, device):
        super().__init__(tensor_request_by_name)
        self.session = requests.Session()
        self.device = device

    def get_meta_tensor(self, weight):
        return super().__getitem__(weight)[0]

    def __getitem__(self, weight):
        tensor, request = super().__getitem__(weight)
        chunk_size = 1024*128
        with tqdm.tqdm(desc=weight, leave=False, total=tensor.nbytes) as pbar:
            #if tensor.nbytes > psutil.virtual_memory().available / 2:
            #    print(weight, 'is more than half available vram, mmapping a file ...')
            #else:

                # we could also further shard the embeddings and lm_head
                # since embeddings are sparsely used
            assert tensor.nbytes < psutil.virtual_memory().available / 2
            buffer = memoryview(bytearray(tensor.nbytes))

            with self.session.send(request, stream=True) as response:
                while pbar.n < pbar.total:
                    pbar.update(
                        response.raw.readinto(
                            buffer[ pbar.n : pbar.n + chunk_size ]
                        )
                    )

        result = torch.frombuffer(
            buffer,
            dtype = tensor.dtype,
            count = tensor.numel(),
            requires_grad = False,
            device = self.device,
        ).reshape(tensor.shape)

        return result

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

    @staticmethod
    def tensor_request_from_json(url, N, data):
        dtype = data['dtype']
        dtype = dict(
            F32 = torch.float32,
            F8_E4M3 = torch.float8_e4m3fn,
            BF16 = torch.bfloat16,
        )[dtype]
        shape = data['shape']
        tensor = torch.empty(shape, dtype=dtype, device='meta')
        start, end = data['data_offsets']
        start += N + 8
        end += N + 8 - 1
        request = requests.Request('GET', url, dict(Range='bytes='+str(start)+'-'+str(end)))
        request = request.prepare()
        return [tensor, request]

    @classmethod
    def from_user_repo_branch(cls, user, repo, branch, device):
        base_url = f'https://huggingface.co/{user}/{repo}/raw/{branch}/'
        lfs_base_url = f'https://huggingface.co/{user}/{repo}/resolve/{branch}/'
    
        safetensors_index_url = base_url + 'model.safetensors.index.json'
    
        print(safetensors_index_url)
    
        with requests.get(safetensors_index_url, stream=True) as response:
            safetensors_index = json.load(response.raw)
    
        print(safetensors_index['metadata'])
    
        fn_by_weight = safetensors_index['weight_map']
    
        urls = [lfs_base_url + fn for fn in set(fn_by_weight.values())]
    
        #url_range_dict = {}
        #data_by_weight = {}

        tensor_request_by_name = {}
    
        with tqdm.tqdm(urls,desc='constructing tensor urls') as pbar:
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
                #dtype = data['dtype']
                #shape = data['shape']
                #start, end = data['data_offsets']
                #start += headersize + 8
                #end += headersize + 8
                #data_by_weight[weight] = data | {'url':url,'N':headersize}
                tensor_request_by_name[weight] = cls.tensor_request_from_json(url, N, data)
        return cls(tensor_request_by_name, device)

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

    return transformers.pipeline('text-generation', model=model, config=config, tokenizer=tokenizer)

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
pipe('Once upon a time,')
