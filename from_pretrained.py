import contextlib
import huggingface_hub, diffusers, transformers, accelerate
import torch
import nettensors

nettensors_kwparams = dict(trust_sizes=True, mem_usage_frac=0.25, disk_usage_frac=0.95)

@classmethod
def diffusers_ModelMixin_from_pretrained(cls, name, torch_dtype=None, low_cpu_mem_usage=None, device_map=None, subfolder='', max_memory=None, offload_folder=None, offload_state_dict=False, use_safetensors=None):
    config = cls.load_config(name, subfolder=subfolder)
    pq = config.get('quantization_config')
    if pq is not None:
        hf_q = diffusers.DiffusersAutoQuantizer.from_config(pq, pq=True)
        hf_q.validate_environment(torch_dtype=torch_dtype)
        torch_dtype = hf_q.update_torch_dtype(torch_dtype)
        low_cpu_mem_usage = True
    else:
        hf_q = None
    with accelerate.init_empty_weights():
        model = cls.from_config(config, low_cpu_mem_usage=low_cpu_mem_usage)
    #state_dict = nettensors.from_hf_hub(name, lfs_filename='diffusion_pytorch_model.safetensors', **nettensors_kwparams)
    state_dict = nettensors.from_hf_hub(name, subfolder=subfolder, **nettensors_kwparams)
    model._convert_deprecated_attention_blocks(state_dict)
    diffusers.models.modeling_utils.load_model_dict_into_meta(model, state_dict, device='cpu', dtype=torch_dtype, model_name_or_path=name, hf_quantizer=hf_q)
    if hf_q is not None:
        hf_q.postprocess_model(model)
        model.hf_quantizer = hf_q
    model.register_to_config(_name_or_path=name)
    model.eval()
    return model

_transformers_PreTrainedModel_from_pretrained = transformers.PreTrainedModel.from_pretrained.__func__
@classmethod
def transformers_PreTrainedModel_from_pretrained(cls, model_id, revision=None, config_patches = {}, subfolder=None, **kwparams):
    config = transformers.AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=True, subfolder=subfolder)
    for key, val in config_patches.items():
        setattr(config, key, val)
    state_dict = nettensors.from_hf_hub(model_id, revision=revision or 'main', subfolder=subfolder, **nettensors_kwparams)
    with transformers.modeling_utils.no_init_weights(), accelerate.init_empty_weights():
        model = _transformers_PreTrainedModel_from_pretrained(cls, None, config=config, state_dict=state_dict, revision=revision, **kwparams)
    return model

_diffusers_DiffusionPipeline_from_pretrained = diffusers.DiffusionPipeline.from_pretrained.__func__
@classmethod
def diffusers_DiffusionPipeline_from_pretrained(cls, name, torch_dtype=None,provider=None,sess_options=None,device_map=None,max_memory=None,offload_folder=None,offload_state_dict=False,from_flax=False,variant=None,low_cpu_mem_usage=diffusers.models.modeling_utils._LOW_CPU_MEM_USAGE_DEFAULT,use_safetensors=None,**kwparams):
    cache_dir = huggingface_hub.snapshot_download(name, allow_patterns=[])
    config = cls.load_config(name)
    config.pop('_ignore_files', None)
    pipeline_class = diffusers.pipelines.pipeline_utils._get_pipeline_class(cls, config=config)
    expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
    init_dict, *_ = pipeline_class.extract_init_dict(config)
    sub_models = {}
    for sub_name, [library_name, class_name] in sorted(init_dict.items()):
        importable_classes = diffusers.pipelines.pipeline_utils.ALL_IMPORTABLE_CLASSES
        is_pipeline_module = hasattr(diffusers.pipelines, library_name)
        class_obj, class_candidates = diffusers.pipelines.pipeline_loading_utils.get_class_obj_and_candidates(
            library_name, class_name, importable_classes, diffusers.pipelines,
            is_pipeline_module, component_name=sub_name, cache_dir=cache_dir)
        load_method = class_obj.from_pretrained
        loading_kwparams = {}
        if issubclass(class_obj, torch.nn.Module):
            loading_kwparams['torch_dtype'] = torch_dtype
        if issubclass(class_obj, diffusers.OnnxRuntimeModel):
            loading_kwparams['provider'] = provider
            loading_kwparams['sess_options'] = sess_options
        is_diffusers_model = issubclass(class_obj, diffusers.ModelMixin)
        is_transformers_model = issubclass(class_obj, transformers.PreTrainedModel)
        if is_diffusers_model or is_transformers_model:
            loading_kwparams.update(dict(
                device_map = device_map,
                max_memory = max_memory,
                offload_folder = offload_folder,
                offload_state_dict = offload_state_dict,
                use_safetensors = use_safetensors
            ))
            if from_flax:
                loading_kwparams['from_flax'] = True
            if not (from_flax and is_transformers_model):
                loading_kwparams['low_cpu_mem_usage'] = low_cpu_mem_usage
            else:
                loading_kwparams['low_cpu_mem_usage'] = False
        loaded_sub_model = load_method(name, **loading_kwparams, subfolder=sub_name)
        sub_models[sub_name] = loaded_sub_model
    return _diffusers_DiffusionPipeline_from_pretrained(cls, name, torch_dtype=torch_dtype,provider=provider,sess_options=sess_options,device_map=device_map,max_memory=max_memory,offload_folder=offload_folder,offload_state_dict=offload_state_dict,from_flax=from_flax,variant=variant,low_cpu_mem_usage=low_cpu_mem_usage,use_safetensors=use_safetensors,**kwparams,**sub_models)

@classmethod
def stub_from_pretrained(cls, *params, **kwparams):
    raise NotImplementedError(cls.__name__ + '.from_pretrained')

@contextlib.contextmanager
def nettensors_from_pretrained(**kwparams):
    if kwparams:
        global nettensors_kwparams
        nettensors_kwparams = kwparams
    stashed_from_pretrained = {}
    try:
        for pkgname, pkg in [['diffusers',diffusers],['transformers',transformers]]:
            for clsname in diffusers.pipelines.pipeline_utils.LOADABLE_CLASSES[pkgname]:
                f = globals().get(f'{pkgname}_{clsname}_from_pretrained')
                if f is None:
                    if 'Pipeline' in clsname or 'Tokenizer' in clsname or 'Scheduler' in clsname:
                        continue
                    f = stub_from_pretrained
                cls = getattr(pkg, clsname)
                stashed_from_pretrained[cls] = getattr(cls, 'from_pretrained')
                setattr(cls, 'from_pretrained', f)
        print(stashed_from_pretrained.keys())
        yield
    finally:
        for cls, stashed in stashed_from_pretrained.items():
            setattr(cls, 'from_pretrained', stashed)
