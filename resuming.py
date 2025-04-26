# the biggest current problem with Resumer is that it doesn't store data from being midway through NetTensor's partial operations


# we can make a torch model resumable when execution is interrupted,
# by recording the outputs between layers and replaying them on next run.
# this is efficient because only the uppermost completed layers need be stored.

# meanwhile, resuming a generation call in transformers means simple exchanges like the shorter inputs for the longer recorded ones
# it ends up being conditions and wrappers or such for however the user caused the model call.
# code could recognize the untokenized string, for example, based on initial inputs inside a resuming context manager.

class Resumer:
    # there is 1 resume state which is where the pass is.
    # it is updated in every pass after it
    # and referenced for data in every pass prior.
    def __init__(self, path, module, name=None, sync_seconds=60*5):
        if name is None:
            name = module.__class__.__name__
        self.resumedata = ResumeData(path, sync_seconds)
        self.module = module
    def _wrap_forward(self, mod, name, index):
        wrapped_forward = mod.forward
        mod.forward = lambda *params, **kwparams: self._forward(mod, name, index, wrapped_forward, params, kwparams)
        return wrapped_forward
    def _forward(self, mod, name, index, forward, params, kwparams):
        #print('\n_forward', name, 'beginning')
        state, data = self.resumedata.module_starting(name, index, [list(params), kwparams])
        if state == self.resumedata.STATE_PROCESSING:
            # first data was input, calculate and stash output
            data = forward(*params, **kwparams)
            state, data = self.resumedata.module_completed(name, index, data)
        #print('\n_forward', name, 'ending')
        return data
    def __enter__(self):
        self.resumedata.__enter__()
        for name, mod in self.module.named_modules():
            mod.__wrapped_forward = self._wrap_forward(mod, name, -1)
        self.module.__wrapped_forward # ensure root module has a hook
    def __exit__(self, *params, **kwparams):
        for name, mod in self.module.named_modules():
            mod.forward = mod.__wrapped_forward
            del mod.__wrapped_forward
        return self.resumedata.__exit__(*params, **kwparams)

import json, os, time
import filelock, torch

def json_serialize(obj):
    if type(obj) is torch.Tensor:
        return {
            '__type': 'torch.Tensor',
            'list': obj.tolist(),
            'dtype': str(obj.dtype).rsplit('.',1)[-1],
            'device': {'type': obj.device.type, 'index': obj.device.index},
        }
    else:
        return obj

def json_deserialize(obj):
    __type = obj.get('__type')
    if __type == 'torch.Tensor':
        return torch.tensor(
            obj['list'],
            dtype = getattr(torch, obj['dtype']),
            device = torch.device(**obj['device']),
        )
    else:
        return obj

class ResumeData:
    STATE_PROCESSING = 0
    STATE_COMPLETED = 1
    def __init__(self, path, sync_seconds=60*5):
        self.path = path
        self.lock = filelock.FileLock(path + '.lock', blocking=False)
        self.sync_seconds = sync_seconds
        self.timestamp = time.time()
    def module_starting(self, name, index, input):
        if self.next_index == len(self.states):
            state = [name, index, self.STATE_PROCESSING, input]
            self.states.append(state)
            self.next_index += 1
            if time.time() > self.timestamp + self.sync_seconds:
                self._sync()
            return state[2:]
        else:
            stored_name, stored_index, stored_state, stored_data = self.states[self.next_index]
            assert stored_name == name and stored_index == index
            if stored_state == self.STATE_PROCESSING:
                torch.testing.assert_close(stored_data, input)
            self.next_index += 1
            return [stored_state, stored_data]
    def module_completed(self, name, index, output):
        assert self.next_index == len(self.states)
        for idx in range(len(self.states)-1,-1,-1):
            stored_state = self.states[idx]
            if stored_state[2] == self.STATE_PROCESSING:
                assert stored_state[:2] == [name, index]
                stored_state[2:] = [self.STATE_COMPLETED, output]
                self.states[idx+1:] = []
                self.next_index = idx + 1
                if time.time() > self.timestamp + self.sync_seconds:
                    self._sync()
                return stored_state[2:]
        else:
            assert not 'no module to complete found'
    def __enter__(self):
        self.lock.__enter__()
        try:
            with open(self.path, 'rt') as f:
                self.states = json.load(f, object_hook=json_deserialize)
        except:
            self.states = []
        self.next_index = 0
        return self
    def _sync(self):
        timestamp = time.time()
        with open(self.path + '.new', 'wt') as f:
            json.dump(self.states, f, default=json_serialize)
        os.rename(self.path + '.new', self.path)
        self.timestamp = timestamp
    def __exit__(self, *params, **kwparams):
        self._sync()
        #del self.states
        return self.lock.__exit__(*params, **kwparams)
