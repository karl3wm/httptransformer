
# this useful module is not presently used yet by httptransformer.py,
# but functions as an independent safetensors-compatible interface to remote safetensors
# and automatically caches to memory-mapped sparse files on-disk respecting a customizable free space ratio
#
# see "if __name__ == '__main__':" block at bottom for usage
#
# note this does not implement network memory mapping: if this is naively plugged in to transformers it will simply download all the tensors into the cache

import codecs, json, math, os, psutil
import torch, transformers, tqdm

import requests, mmap

class RequestsFetchers:
    session = requests.Session()
    def __init__(self, base_url, headers={}):
        self._base_url = base_url
        self._headers = headers
        self.fetchers = {}
    def fetcher(self, filename):
        fetcher = self.fetchers.get(filename)
        if fetcher is None:
            fetcher = self.Fetcher(filename, self)
            self.fetchers[filename] = fetcher
        return fetcher
    class Fetcher:
        def __init__(self, filename, fetchers = None):
            if fetchers is None:
                base_url, filename = filename.rsplit('/',1)
                fetchers = RequestsFetchers(base_url)
            self.filename = filename
            self.fetchers = fetchers
            self.session = fetchers.session
            url = f'{fetchers._base_url}/{filename}'
            headers = fetchers._headers
            self.request = requests.Request('GET', url, headers=headers).prepare()
            self.head_request = requests.Request('HEAD', url, headers=headers).prepare()
        def read(self, offset, length, progress:str=''):
            assert length > 0
            self.request.prepare_headers(dict(**self.fetchers._headers,Range=f'bytes={offset}-{offset+length-1}'))
            response = self.session.send(self.request, stream=True)
            buf = memoryview(bytearray(length))
            if progress:
                with tqdm.tqdm(desc=progress, total=length, unit='B', unit_scale=True, unit_divisor=1024, leave=False) as pbar:
                    while pbar.n < length:
                        readsize = response.raw._fp.readinto(buf[pbar.n:pbar.n+1024*256])
                        assert readsize
                        pbar.update(readsize)
            else:
                readsize = response.raw.readinto(buf)
                assert readsize == length
            return buf.obj
        def size(self):
            return int(self.session.send(self.head_request).headers['Content-Length'])
        def __str__(self):
            return self.request.url

class CachedRequestsFetchers(RequestsFetchers):
    sparse_usage = 0
    warned_space = False
    def __init__(self, folder, base_url, headers={}, usage_frac=0.5, trust_sizes=False):
        super().__init__(base_url, headers=headers)
        self.folder = folder
        self.usage_frac = usage_frac
        self.trust_sizes = trust_sizes
        os.makedirs(folder, exist_ok=True)
    class Fetcher(RequestsFetchers.Fetcher):
        def __init__(self, filename, fetchers=None):
            super().__init__(filename, fetchers)
            self.fn = os.path.join(self.fetchers.folder, self.request.url.rsplit('/',1)[-1])
            size = 0
            if os.path.exists(self.fn):
                size = os.stat(self.fn).st_size
            if size == 0:
                size = super().size()
            self.fd = os.open(self.fn, os.O_RDWR | os.O_CREAT)
            try:
                self.mmap = memoryview(mmap.mmap(self.fd, size))
            except ValueError:
                with open(self.fn, 'wb+') as fh:
                    fh.truncate(size)
                self.mmap = memoryview(mmap.mmap(self.fd, size))
            self.blksize = math.lcm(os.statvfs(self.fn).f_bsize, os.sysconf('SC_PAGESIZE'), mmap.PAGESIZE)

            if not self.fetchers.trust_sizes:
                assert self.size() == super().size()

            # measure usage
            cls = type(self.fetchers)
            data_off = 0
            while data_off < size:
                hole_off = self._next_sparse(data_off, os.SEEK_HOLE)
                cls.sparse_usage += hole_off - data_off
                data_off = self._next_sparse(hole_off, os.SEEK_DATA)

        def read(self, offset, length, progress:str=''):
            tail = min(offset + length, len(self.mmap))
            aligned_offset = (offset // self.blksize) * self.blksize
            aligned_tail = min(self.size(), (((tail - 1) // self.blksize) + 1) * self.blksize)

            next_hole = self._next_sparse(aligned_offset, os.SEEK_HOLE)

            if next_hole < tail:
                # data not cached
                cls = type(self.fetchers)
                if cls.sparse_usage + aligned_tail - aligned_offset > (psutil.disk_usage(self.fn).free + cls.sparse_usage) * self.fetchers.usage_frac:
                    # no more disk space
                    if not cls.warned_space:
                        import warnings
                        warnings.warn(
                            '\nCACHE FULL CACHE FULL' +
                            '\nRequested=' +
                            str(tqdm.tqdm.format_sizeof(aligned_tail - aligned_offset, 'B', 1024))
                            + ' Cached=' +
                            str(tqdm.tqdm.format_sizeof(cls.sparse_usage, 'B', 1024))
                            + ' Free=' +
                            str(tqdm.tqdm.format_sizeof(psutil.disk_usage(self.fn).free, 'B', 1024))
                            + '\n' +
                            os.path.dirname(self.fn)
                            + '\nCACHE FULL CACHE FULL'
                        , stacklevel=5)
                        cls.warned_space = True
                    return super().read(offset, length, progress=progress)

                hole_on_left = self._next_sparse(aligned_offset - 1, os.SEEK_HOLE) < aligned_offset

                next_data = self._next_sparse(next_hole, os.SEEK_DATA)
                while next_data < tail:
                    assert next_data - next_hole <= length
                    length = next_data - next_hole
                    self.mmap[next_hole:next_data] = super().read(next_hole, length, progress=progress)
                    cls.sparse_usage += length
                    next_hole = self._next_sparse(next_data, os.SEEK_HOLE)
                    next_data = self._next_sparse(next_hole, os.SEEK_DATA)
                if next_hole < tail:
                    length = aligned_tail - next_hole
                    self.mmap[next_hole:aligned_tail] = super().read(next_hole, length, progress=progress)
                    cls.sparse_usage += length
                    # updated this while sleepy
                    # on docker vms i found the memory mapper filling extra blocks with 0s
                    # this new code tries to ensure data is correct when that happens
                    # i've also updated the pagesize calculation so this might happen less
                    extra_0s_right = min(self._next_sparse(aligned_tail, os.SEEK_HOLE), next_data)
                    while extra_0s_right > aligned_tail:
                        length = extra_0s_right - aligned_tail
                        self.mmap[aligned_tail:extra_0s_right] = super().read(aligned_tail, length, progress=progress)
                        cls.sparse_usage += length
                        extra_0s_right = min(self._next_sparse(aligned_tail, os.SEEK_HOLE), next_data)
                if hole_on_left:
                    if self._next_sparse(aligned_offset - 1, os.SEEK_HOLE) >= aligned_offset:
                        # the hole on the left disappeared
                        # this could be resolved by walking holes on the left or storing auxiliary data regarding allocated regions
                        # the former is space efficient and the latter time efficient; they could be combined as well
                        os.unlink(self.fn)
                        raise Exception(
                            'Your memory mapper is writing data below the cached region ' +
                            'even when aligned to the pagesize and blocksize. ' +
                            'The current code generates corrupt cached runs of 0s in this situation.')
            return self.mmap[offset:tail]

        def size(self):
            return len(self.mmap)

        def _next_sparse(self, off, region):
            try:
                return os.lseek(self.fd, off, region)
            except OSError as err:
                if err.errno == 6: # ENXIO, no such file address
                    return len(self.mmap)
                raise

class SafeSlice:
    def __init__(self, offset, tensor, fetcher):
        self.offset = offset
        self.tensor = tensor
        self.fetcher = fetcher
    def get_shape(self):
        return self.tensor.shape
    def get_dtype(self):
        return self.tensor.dtype
    def __getitem__(self, slice):
        tensor = self.tensor[slice]
        offset = self.offset + tensor.storage_offset() * tensor.element_size()
        data = self.fetcher.read(offset, tensor.nbytes)
        return torch.frombuffer(
            data,
            dtype=tensor.dtype,
            count=tensor.numel(),
        ).view(tensor.shape)

class SafeTensors:
    def __init__(self, fetcher):
        if type(fetcher) is str:
            fetch = RequestsFetchers.Fetcher(fetcher)
        self.fetcher = fetcher

        N_encoded = self.fetcher.read(0, 8)
        assert len(N_encoded) == 8
        N = int.from_bytes(N_encoded, 'little')
        assert N <= self.fetcher.size() - 8
        header = json.loads(codecs.decode(self.fetcher.read(8, N)))
        tensors = {}
        for name, data in tqdm.tqdm(header.items(), desc=str(fetcher), leave=False):
            if name == '__metadata__':
                self.__metadata__ = data
            else:
                start, end = data['data_offsets']
                offset = start + N + 8
                dtype = data['dtype'].replace('F', 'float').replace('I', 'int').lower()
                try:
                    dtype = getattr(torch, dtype)
                except:
                    dtype = getattr(torch, dtype + 'fn')
                tensor = torch.empty(data['shape'], dtype=dtype, device='meta')
                assert tensor.nbytes == (end - start)
                tensors[name] = [offset, tensor]
        self.__tensors__ = tensors
    def get_slice(self, name):
        offset, tensor = self.__tensors__[name]
        return SafeSlice(offset, tensor, self.fetcher)
    def get_tensor(self, name):
        return self.get_slice(name)[:]
    def keys(self):
        return self.__tensors__.keys()
    def metadata(self):
        return self.__metadata__

class SafeTensorsIndex:
    def __init__(self, fetchers, filename = transformers.utils.SAFE_WEIGHTS_INDEX_NAME):
        if type(fetchers) is str:
            fetchers = RequestsFetchers(fetchers)
        self.fetchers = fetchers
        index = self.fetchers.fetcher(filename)
        index = json.loads(codecs.decode(index.read(0, index.size(), progress=filename)))
        self.__metadata__ = index['metadata']
        self.__tensors__ = {
            name : self.fetchers.fetcher(fn)
            for name, fn in tqdm.tqdm(index['weight_map'].items(), desc='Checking tensor cache sizes', unit='name', leave=False)
        }
    def get_slice(self, name):
        return self.get_safetensors(name).get_slice(name)
    def get_tensor(self, name):
        return self.get_safetensors(name).get_tensor(name)
    def keys(self):
        return self.__tensors__.keys()
    def metadata(self):
        return self.__metadata__
    def get_safetensors(self, name):
        safetensors = self.__tensors__[name]
        if type(safetensors) is self.fetchers.Fetcher:
            safetensors = SafeTensors(safetensors)
            self.__tensors__[name] = safetensors
        return safetensors

def _hf_cache_lfs_fetchers(repo_id, revision, repo_type, **kwparams):
    import huggingface_hub
    for test_fn in ['README.md'] + [fn for fn in transformers.utils.__dict__.values() if type(fn) is str and fn.endswith('.json')]:
        try:
            folder = os.path.dirname(huggingface_hub.hf_hub_download(repo_id, test_fn, revision=revision, repo_type=repo_type))
            break
        except:
            pass
    repo_path = {
            None: '',
            'model': '',
            'dataset': 'datasets/',
            'space': 'spaces/'
    }[repo_type] + repo_id
    return CachedRequestsFetchers(
        os.path.join(folder, 'netsafetensors'),
        f'https://huggingface.co/{repo_path}/resolve/{revision}',
        headers = huggingface_hub.utils.build_hf_headers(),
        **kwparams
    )

def from_hf_hub(repo_id, lfs_filename = None, revision='main', repo_type=None, **kwparams):
    fetchers = _hf_cache_lfs_fetchers(repo_id, revision, repo_type, **kwparams)
    if lfs_filename is not None:
        if lfs_filename == transformers.utils.SAFE_WEIGHTS_NAME:
            return SafeTensors(fetchers.fetcher(lfs_filename))
        elif lfs_filename == transformers.utils.SAFE_WEIGHTS_INDEX_NAME:
            return SafeTensorsIndex(fetchers, lfs_filename)
        else:
            try:
                return SafeTensors(fetchers.fetcher(lfs_filename))
            except AssertionError:
                return SafeTensorsIndex(fetchers, lfs_filename)
    else:
        try:
            return SafeTensors(
                fetchers.fetcher(transformers.utils.SAFE_WEIGHTS_NAME)
            )
        except AssertionError:
            return SafeTensorsIndex(fetchers, transformers.utils.SAFE_WEIGHTS_INDEX_NAME)


if __name__ == '__main__':
    safetensors = from_hf_hub('deepseek-ai/DeepSeek-V3')
    keys = list(safetensors.keys())
    sliceable = safetensors.get_slice(keys[0])
    sliceable[0]
    bigger = safetensors.get_tensor(keys[0])
