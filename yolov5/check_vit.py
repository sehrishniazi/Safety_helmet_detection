# check_vit.py — place in yolov5/ and run from that folder
import importlib, torch, sys

try:
    m = importlib.import_module('models.backbone_vit')
    print('Imported models.backbone_vit OK')
    cls = getattr(m, 'SimpleViTBackbone')
    print('Found class SimpleViTBackbone')
    # instantiate with pretrained=False to avoid downloading weights for quick test
    model = cls(model_name='vit_tiny_patch16_224', pretrained=False, out_channels=128)
    model.eval()
    x = torch.randn(1,3,416,416)
    with torch.no_grad():
        feats = model(x)
    print('Returned feature list lengths:', len(feats))
    print('Shapes:', [tuple(f.shape) for f in feats])
except Exception as e:
    print('ERROR:', repr(e))
    sys.exit(1)
