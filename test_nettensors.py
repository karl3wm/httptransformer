import accelerate, torch, transformers
import nettensors

model_id, revision = 'meta-llama/Llama-3.1-405B', 'b906e4dc842aa489c962f9db26554dcfdde901fe'
#model_id, revision = 'Nexusflow/Athene-V2-Chat', '493f1bbd561a5a7e3d27c4081d4ee47508bf6831'
#model_id, revision = 'meta-llama/Llama-3.2-1B', '4e20de362430cd3b72f300e6b0f18e50e7166e08'

import contextlib, functools
class Quirks:
    @contextlib.contextmanager
    def init_empty_weights():
        # accelerate.init_empty_weights replaces register_parameter
        # in a way that prevents model.tie_weights() from succeeding
        # this then leaves dangling unloaded parameters, which
        #  transformers then enumerates and attempts to move on-device
        #  -- but they are empty meta tensors generated by accelerate
        def wrap_tensor_constructor(module, funcname):
            func = getattr(module, funcname)
            @functools.wraps(func)
            def wrapper(*params, **kwparams):
                kwparams['device'] = 'meta'
                return func(*params, **kwparams)
            def remove():
                setattr(module, funcname, func)
            setattr(module, funcname, wrapper)
            return remove
        with transformers.modeling_utils.no_init_weights():
            unwraps = [
                wrap_tensor_constructor(torch, 'empty')
            ]
            try:
                yield
            finally:
                [unwrap() for unwrap in unwraps]

def construct(model_id, revision):
    config = transformers.AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    state_dict = nettensors.from_hf_hub(model_id, revision=revision, trust_sizes=True, mem_usage_frac=0.33, disk_usage_frac=0.95)
    with Quirks.init_empty_weights():
        model = transformers.AutoModelForCausalLM.from_pretrained(None, config=config, state_dict=state_dict, device_map='cpu', torch_dtype=torch.float64)
    return transformers.pipeline('text-generation', model=model, config=config, tokenizer=tokenizer)

print(model_id, revision)
pipe = construct(model_id, revision)

check = nettensors.from_hf_hub('baffo32/llm_logits', repo_type='dataset', lfs_filename=f"{model_id.replace('/','_')}_{revision}.logits.safetensors")

_nested = False
_first_module = None
def compare(module, inputs, output, local, stored):
    distance = (local - stored).abs()
    err = distance.sum() / stored.abs().sum()
    if err > 1.0/8:
        if module.weight.mem_usage_frac() < 1:
            module.weight = torch.nn.Parameter(module.weight.fetch())
        import pdb; pdb.set_trace()
        'local and stored differ significantly'
def hook(module, inputs, output):
    global _nested, _first_module
    if _nested:
        return
    if _first_module is None:
        _first_module = module
    elif _first_module is module:
        _nested = True # no more comparison data, stop checking
        return
    _nested = True
    name = modnames[module]
    for idx, input in enumerate(inputs):
        if type(input) is torch.Tensor:
            cmp = check[f'{name}.input.{idx}'].fetch()
            compare(module, inputs, output, input, cmp)
    if type(output) is torch.Tensor:
        cmp = check[f'{name}.output'].fetch()
        compare(module, inputs, output, output, cmp)
    _nested = False
modnames = {mod:name for name,mod in pipe.model.named_modules()}
[mod.register_forward_hook(hook) for mod in modnames]

class Streamer(transformers.TextStreamer):
    def on_finalized_text(self, text, stream_end=False):
        print(); print(text)

pipe('Once upon a time,', streamer=Streamer(pipe.tokenizer))
