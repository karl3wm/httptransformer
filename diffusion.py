import accelerate, torch, diffusers.utils, diffusers
import nettensors, from_pretrained

#def construct(cls, name, torch_dtype=None, low_cpu_mem_usage=None):
#    config = cls.load_config(name)
#    pq = config.get('quantization_config')
#    if pq is not None:
#        hf_q = diffusers.DiffusersAutoQuantizer.from_config(pq, pq=True)
#        hf_q.validate_environment(torch_dtype=torch_dtype)
#        torch_dtype = hf_q.update_torch_dtype(torch_dtype)
#        low_cpu_mem_usage = True
#    else:
#        hf_q = None
#    with accelerate.init_empty_weights():
#        model = cls.from_config(config, low_cpu_mem_usage=low_cpu_mem_usage)
#    state_dict = nettensors.from_hf_hub(name, lfs_filename='diffusion_pytorch_model.safetensors')
#    model._convert_deprecated_attention_blocks(state_dict)
#    diffusers.models.modeling_utils.load_model_dict_into_meta(model, state_dict, device='cpu', dtype=torch_dtype, model_name_or_path=name, hf_quantizer=hf_q)
#    if hf_q is not None:
#        hf_q.postprocess_model(model)
#        model.hf_quantizer = hf_q
#    #model = model.to(torch_dtype)
#    model.register_to_config(_name_or_path=name)
#    model.eval()
#    return model
#
#def pipeline_modules(cls, name, torch_dtype=None):
#    config = cls.load_config(name)
#    config.pop('_ignore_files', None)
#    pipeline_class = diffusers.pipelines.pipeline_loading_utils._get_pipeline_class(cls, config=config)
#    expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
#    init_dict, *_ = pipeline_class.extract_init_dict(config)
#    for name, [library_name, class_name] in init_dict.items():
#
#    print(init_dict)

base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model_union = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0'

with from_pretrained.nettensors_from_pretrained():
    controlnet = diffusers.FluxControlNetModel.from_pretrained(controlnet_model_union, torch_dtype=torch.bfloat16)
    pipe = diffusers.FluxControlNetPipeline.from_pretrained(base_model, controlnet=[controlnet], torch_dtype=torch.bfloat16)

#controlnet = construct(diffusers.FluxControlNetModel, controlnet_model_union, torch_dtype=torch.bfloat16)
#pipeline_modules(diffusers.FluxControlNetPipeline, base_model, torch_dtype=torch.bfloat16)

