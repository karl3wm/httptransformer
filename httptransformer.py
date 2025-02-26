import json, requests, tqdm

user = 'deepseek-ai'
repo = 'DeepSeek-V3'
branch = 'main'
base_url = f'https://huggingface.co/{user}/{repo}/raw/{branch}/'
lfs_base_url = f'https://huggingface.co/{user}/{repo}/resolve/{branch}/'

safetensors_index_url = base_url + 'model.safetensors.index.json'

print(safetensors_index_url)

with requests.get(safetensors_index_url, stream=True) as response:
    safetensors_index = json.load(response.raw)

print(safetensors_index['metadata'])

fn_by_weight = safetensors_index['weight_map']

urls = [lfs_base_url + fn for fn in set(fn_by_weight.values())]

data_by_weight = {}

with tqdm.tqdm(urls,desc='urls') as pbar:
  for url in pbar:
    # we could potentially also check the git-lfs sha256 from the base url and merklify the data too, this would mean downloading it all
    #[b'version https://git-lfs.github.com/spec/v1', b'oid sha256:e94d32e8649e1a5b03cc0a343c59ca5a6d80d03cd46161b482fd3bb2484adb7d', b'size 4302350824']
    #lfs = dict([ line.decode().split(' ', 1) for line in response.iter_lines() ])
    with requests.get(url, stream=True) as response:
        headersize = int.from_bytes(response.raw.read(8), 'little')
        header = json.loads(response.raw.read(headersize))
    for weight, data in header.items():
        if weight == '__metadata__':
            continue
        data_by_weight[weight] = data | {'url':url,'N':headersize}
print(*list(data_by_weight.items())[0])
