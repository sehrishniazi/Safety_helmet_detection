# test_full_model.py - Complete model test using YOLOv5's Model class
import torch
from models.yolo import Model

# Create model from YAML
print('🏗️  Building model from YAML...')
model = Model('models/custom_vit_yolov5s.yaml', ch=3, nc=3)

print(f'✅ Model created successfully!')
print(f'📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}')

# Test forward pass
print('\n🧪 Testing forward pass with 320x320 input...')
model.eval()
dummy_input = torch.randn(1, 3, 320, 320)

try:
    with torch.no_grad():
        output = model(dummy_input)
    
    print('✅ Forward pass successful!')
    
    if isinstance(output, (list, tuple)):
        print(f'\n📦 Model outputs {len(output)} items:')
        for i, item in enumerate(output):
            if isinstance(item, torch.Tensor):
                print(f'   Output[{i}]: shape {tuple(item.shape)}')
            elif isinstance(item, list):
                print(f'   Output[{i}]: list of {len(item)} tensors')
                for j, t in enumerate(item[:3]):  # Show first 3
                    print(f'      [{j}]: {tuple(t.shape)}')
    else:
        print(f'📦 Output shape: {tuple(output.shape)}')
        
    # Try to understand the output structure (YOLOv5 returns different things in train vs eval)
    print('\n📋 Output interpretation:')
    if isinstance(output, tuple) and len(output) == 2:
        inference_out, train_out = output
        print('   Mode: Evaluation (returns inference + training outputs)')
        print(f'   Inference output: {len(inference_out)} detection layers')
        for i, det in enumerate(inference_out):
            print(f'      P{i+3}: {tuple(det.shape)}')
    elif isinstance(output, list):
        print(f'   Mode: Training (returns {len(output)} feature maps)')
        for i, feat in enumerate(output):
            print(f'      P{i+3}: {tuple(feat.shape)}')
            
except Exception as e:
    print(f'❌ Forward pass failed!')
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()

print('\n✨ Testing complete!')