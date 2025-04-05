# we can make a torch model resumable when execution is interrupted,
# by recording the outputs between layers and replaying them on next run. (see the check hook in test_nettensors.py)
# this is efficient because only the uppermost completed layers need be stored.

# meanwhile, resuming a generation call in transformers means simple exchanges like the shorter inputs for the longer recorded ones
# it ends up being conditions and wrappers or such for however the user caused the model call.
# code could recognize the untokenized string, for example, based on initial inputs inside a resuming context manager.
