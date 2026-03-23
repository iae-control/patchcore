
content = open('compare_distributions_half.py').read()
old = "TRANSFORM = transforms.Compose([\n    transforms.ToTensor(),"
new = "TRANSFORM = transforms.Compose([\n    transforms.Resize((600, 960), interpolation=transforms.InterpolationMode.BILINEAR),\n    transforms.ToTensor(),"
content = content.replace(old, new)
open('compare_distributions_half.py', 'w').write(content)
print('Fixed')
