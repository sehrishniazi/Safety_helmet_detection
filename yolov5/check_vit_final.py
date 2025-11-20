# check_vit_final.py
import importlib, torch, sys
m = importlib.import_module('models.backbone_vit')
cls = getattr(m, 'SimpleViTBackbone')
model = cls(model_name='vit_tiny_patch16_224', pretrained=False, out_channels=256, img_size=320)
model.eval()
x = torch.randn(1,3,320,320)
with torch.no_grad():
    out = model(x)
print('backbone output type:', type(out))
if isinstance(out, (list,tuple)):
    print('len=', len(out), 'shapes=', [tuple(o.shape) for o in out])
else:
    print('shape=', tuple(out.shape))
