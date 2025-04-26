# we can make a torch model resumable when execution is interrupted,
# by recording the outputs between layers and replaying them on next run. (see the check hook in test_nettensors.py)
# this is efficient because only the uppermost completed layers need be stored.

# meanwhile, resuming a generation call in transformers means simple exchanges like the shorter inputs for the longer recorded ones
# it ends up being conditions and wrappers or such for however the user caused the model call.
# code could recognize the untokenized string, for example, based on initial inputs inside a resuming context manager.

# I WORKED ON THE BELOW FOR FUN WHEN IT WAS REALLY HARD
# FEEL FREE TO RESTART / WIPE 

class ResumeModule:
    # there is 1 resume state which is where the pass is.
    # it is updated in every pass after it
    # and referenced for data in every pass prior.
    def __init__(self, module, path, seconds=60*5):
        self.path = path
        self.state = {}
        for name, mod in module.named_modules():
            
            module.register_forward_hook(self.__hook)
            mod.__name = name
            #self.state [mod.
    def __enter__(self):

    def __exit__(self, *params, **kpwarams):
        return super().__exit__(*params, **kwparams)
    def _forward_store_resume(self, modid, wrapped, *params, **kwparams):
        # two cases: we are performing a resume,
        # or we are storing resume state.
        # the second is more common. the first could have its own hooks.
        result = wrapped(*params, **kwparams)
        return result

# ideas for replacing modules within torch spec:
#   - register_[module_]forward_pre_hook could change the module's behavior, and then another hook put it back after
#   - manual replacement of forward function, might be similar
#   - inspection of __call__ source to see default space clearly
#       __call__ calls _call_impl which uses local variables like forward_call.
#       _call_impl(self, *args, **kwargs)



