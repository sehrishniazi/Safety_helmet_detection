# debug_parse.py
import yaml
from copy import deepcopy
from models.yolo import parse_model

cfg = yaml.safe_load(open('models/custom_vit_yolov5s.yaml'))
# parse_model expects the model dict and ch list; ch=[3] means input channels
model, save = parse_model(deepcopy(cfg), ch=[3])
print('Parsed model layers (index, from, n, module, args):')
for i, m in enumerate(model):
    from_idxs, n, module, args = m['from'], m['n'], m['module'].__class__.__name__, m.get('args', None)
    print(f'{i:2d}  from={from_idxs}  n={n:2d}  module={module}  args={args}')
