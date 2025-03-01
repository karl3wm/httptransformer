
# this useful module is not presently used yet by httptransformer.py,
# but functions as an independent safetensors-compatible interface to remote safetensors

import json, os, psutil
import torch, tqdm

class RequestsFetcher:
    session = None
    def __init__(self, url, session=None):
        import requests
        if session is None:
            if self.session is None:
                type(self).session = requests.Session()
        else:
            self.session = session
        self.request = requests.Request('GET', url).prepare()
    def read(self, offset, length):
        assert length > 0
        self.request.prepare_headers(dict(Range=f'bytes={offset}-{offset+length-1}'))
        return self.session.send(self.request).content
    def size(self):
        return int(self.session.head(self.request.url, allow_redirects=True).headers['Content-Length'])
    def __str__(self):
        return self.request.url

class CachedRequestsFetcher(RequestsFetcher):
    sparse_usage = 0
    warned_space = False
    def __init__(self, folder, url, session=None, usage_frac=0.5):
        super().__init__(url, session)
        import mmap
        os.makedirs(folder, exist_ok=True)
        self.fn = os.path.join(folder, url.rsplit('/',1)[-1])
        size = 0
        if os.path.exists(self.fn):
            size = os.stat(self.fn).st_size
        if size == 0:
            size = super().size()
        self.usage_frac = usage_frac
        self.fd = os.open(self.fn, os.O_RDWR | os.O_CREAT)
        try:
            self.mmap = mmap.mmap(self.fd, size)
        except ValueError:
            with open(self.fn, 'wb+') as fh:
                fh.truncate(size)
            self.mmap = mmap.mmap(self.fd, size)
        self.blksize = os.statvfs(self.fn).f_bsize
        assert self.size() == super().size()

        # measure usage
        cls = type(self)
        data_off = 0
        while data_off < size:
            hole_off = self._next_sparse(data_off, os.SEEK_HOLE)
            cls.sparse_usage += hole_off - data_off
            data_off = self._next_sparse(hole_off, os.SEEK_DATA)

    def read(self, offset, length):
        next_hole = self._next_sparse(offset, os.SEEK_HOLE)
        tail = min(offset + length, len(self.mmap))
        if next_hole < tail:
            # data not cached
            if self.sparse_usage + length > (psutil.disk_usage(self.fn).free + self.sparse_usage) * self.usage_frac:
                # no more disk space
                if not self.warned_space:
                    import warnings
                    warnings.warn('Reached disk usage fraction. Fetching new data live from the web.', stacklevel=3)
                    type(self).warned_space = True
                return super().read(offset, length)
            next_data = self._next_sparse(next_hole, os.SEEK_DATA)
            while next_data < tail:
                assert next_data - next_hole <= length
                self.mmap[next_hole:next_data] = super().read(next_hole, next_data - next_hole)
                next_hole = self._next_sparse(next_data, os.SEEK_HOLE)
                next_data = self._next_sparse(next_hole, os.SEEK_DATA)
            if next_hole < tail:
                aligned_tail = (((tail - 1) // self.blksize) + 1) * self.blksize
                self.mmap[next_hole:aligned_tail] = super().read(next_hole, aligned_tail - next_hole)
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
    def __getitem__(self, slice):
        tensor = self.tensor[slice]
        offset = self.offset + tensor.storage_offset() * tensor.element_size()
        data = self.fetcher.read(offset, tensor.nbytes)
        return torch.frombuffer(
            data,
            dtype=tensor.dtype,
            count=tensor.numel()
        ).view(tensor.shape)

class SafeTensors:
    def __init__(self, fetcher):
        if type(fetcher) is str:
            fetch = RequestsFetcher(str)
        self.fetcher = fetcher

        N = int.from_bytes(self.fetcher.read(0, 8), 'little')
        header = json.loads(self.fetcher.read(8, N))
        tensors = {}
        for name, data in tqdm.tqdm(header.items(), desc=str(fetcher)):
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

def from_hf_hub(repo_id, lfs_filename, revision='main', repo_type=None):
    import huggingface_hub
    try:
        folder = os.path.dirname(huggingface_hub.hf_hub_download(repo_id, 'config.json', revision=revision, repo_type=repo_type))
    except:
        folder = os.path.dirname(huggingface_hub.hf_hub_download(repo_id, 'README.md', revision=revision, repo_type=repo_type))
    repo_path = {
            None: '',
            'model': '',
            'dataset': 'datasets/',
            'space': 'spaces/'
    }[repo_type] + repo_id
    fetcher = CachedRequestsFetcher(
        os.path.join(folder, 'netsafetensors'),
        f'https://huggingface.co/{repo_path}/resolve/{revision}/{lfs_filename}'
    )
    return SafeTensors(fetcher)

if __name__ == '__main__':
    st = from_hf_hub('deepseek-ai/DeepSeek-V3', 'model-00140-of-000163.safetensors')
