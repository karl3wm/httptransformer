import accelerate, torch, transformers
import nettensors

import contextlib, functools
class Quirks:
    @contextlib.contextmanager
    def init_empty_weights():
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
        with transformers.modeling_utils.no_init_weights(), accelerate.init_on_device(torch.device('meta')):
            unwraps = [
                wrap_tensor_constructor(torch, 'empty')
            ]
            try:
                yield
            finally:
                [unwrap() for unwrap in unwraps]

def construct(model_id):
    config = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    state_dict = nettensors.from_hf_hub(model_id, trust_sizes=True)
    with Quirks.init_empty_weights():
        model = transformers.AutoModelForCausalLM.from_pretrained(None, config=config, state_dict=state_dict, device_map='cpu')
    return transformers.pipeline('text-generation', model=model, config=config, tokenizer=tokenizer)

import huggingface_hub
pipe = construct('meta-llama/Llama-3.1-405B')
print(pipe('Once upon a time,'))
