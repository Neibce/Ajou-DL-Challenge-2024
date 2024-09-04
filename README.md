# VLM-Ajou-Challenge-2024

- 2024 소프트웨어 중심 대학 공동 딥러닝 챌린지 (24.07.26. ~ 24.08.30.)
- Public, Private score 1st
- Zero-shot classification

![private](https://github.com/user-attachments/assets/0fc8d319-017d-42a2-821d-f7bcf985f24a)

## Pre-Processing
- [RandomAdjustSharpness](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomAdjustSharpness.html)(2, p=1)
```python
processor_20 = T.Compose(
    [
         T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
         T.CenterCrop(size=(224, 224)),
         T.Lambda(lambda img: img.convert('RGB')),
         RandomAdjustSharpness(2, p=1),
         T.ToTensor(),
         T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
     ]
)
```

- [RandomAdjustSharpness](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomAdjustSharpness.html)(2.3, p=1)
```python
processor_23 = T.Compose(
    [
         T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
         T.CenterCrop(size=(224, 224)),
         T.Lambda(lambda img: img.convert('RGB')),
         RandomAdjustSharpness(2.3, p=1),
         T.ToTensor(),
         T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
     ]
)
```

## Prompt-Tuning
- 주요 프롬프트(1%)
```python
    prompts.append(f"a blurry photo of a {class_name}")
    prompts.append(f"a blurry image of a {class_name}")
    prompts.append(f"a blurred photo of a {class_name}")
    prompts.append(f"a blurred image of a {class_name}")
```

- 전체
```python
prompts = []
for class_name in class_names:
    prompts.append(f"{class_name}")
    prompts.append(f"a photo of a {class_name}")
    prompts.append(f"a image of a {class_name}")
    prompts.append(f"art of the {class_name}")
    prompts.append(f"a blurry photo of a {class_name}")
    prompts.append(f"a blurry image of a {class_name}")
    prompts.append(f"a blurred photo of a {class_name}")
    prompts.append(f"a blurred image of a {class_name}")
    prompts.extend([
        f'a bad photo of a {class_name}.',
        f'a photo of many {class_name}.',
        f'a photo of the hard to see {class_name}.',
        f'a low resolution photo of the {class_name}.',
        f'a bad photo of the {class_name}.',
        f'a cropped photo of the {class_name}.',
        f'a photo of a hard to see {class_name}.',
        f'a bright photo of a {class_name}.',
        f'a photo of a clean {class_name}.',
        f'a photo of a dirty {class_name}.',
        f'a dark photo of the {class_name}.',
        f'a photo of my {class_name}.',
        f'a photo of the cool {class_name}.',
        f'a bright photo of the {class_name}.',
        f'a cropped photo of a {class_name}.',
        f'a photo of the dirty {class_name}.',
        f'a jpeg corrupted photo of a {class_name}.',
        f'a blurry photo of the {class_name}.',
        f'a photo of the {class_name}.',
        f'a good photo of the {class_name}.',
        f'a rendering of the {class_name}.',
        f'a {class_name} in a video game.',
        f'a photo of one {class_name}.',
        f'a close-up photo of the {class_name}.',
        f'the {class_name} in a video game.',
        f'a sketch of a {class_name}.',
        f'a low resolution photo of a {class_name}.',
        f'a photo of the clean {class_name}.',
        f'a photo of a large {class_name}.',
        f'a photo of a nice {class_name}.',
        f'a photo of a weird {class_name}.',
        f'a sketch of the {class_name}.',
        f'a jpeg corrupted photo of the {class_name}.',
        f'a good photo of a {class_name}.',
        f'a photo of the nice {class_name}.',
        f'a photo of the small {class_name}.',
        f'a photo of the weird {class_name}.',
        f'a drawing of the {class_name}.',
        f'a photo of the large {class_name}.',
        f'a dark photo of a {class_name}.',
        f'a photo of a small {class_name}.'
    ])

    if class_name == "Buildings":
        prompts.extend([
            "A picture of an urban area with buildings",
            "An architectural structure in the city",
            "The Windows"
        ])
    elif class_name == "Forests":
        prompts.extend([
            "A picture of a dense forest with trees",
            "A scenic view of a forest landscape",
            "A picture of the Trees"
        ])
    elif class_name == "Glacier":
        prompts.extend([
            "A picture of an ice",
            "A scenic view of a snowy glacier",
            "A scenic view of some snow in the mountains"
        ])
    elif class_name == "Mountains":
        prompts.extend([
            "A picture of a mountain range",
            "A scenic view of the mountains",
            "A stunning panorama of rugged mountain cliffs"
        ])
    elif class_name == "Sea":
        prompts.extend([
            "A picture of water",
            "A picture of the ocean",
            "A scenic view of the sea and waves"
        ])
    elif class_name == "Street":
        prompts.extend([
            "A picture of a road",
            "A picture of a busy street in the city",
            "An urban street with buildings and cars"
        ])
```
## Load Pre-Trained Models
### [MetaCLIP](https://github.com/facebookresearch/MetaCLIP) (ViT-bigG-14-quickgelu)
```python
model = open_clip.create_model('ViT-bigG-14-quickgelu', pretrained='metaclip_2_5b').to(device)
```

### [EVA-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP-18B) (EVA-CLIP-18B)
```python
model = AutoModel.from_pretrained(
    'BAAI/EVA-CLIP-18B',
    torch_dtype=torch.float16,
    trust_remote_code=True).to('cuda').eval()
```

## Enassemble
[MetaCLIP](https://github.com/facebookresearch/MetaCLIP)(ViT-bigG-14-quickgelu) + [EVA-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP-18B) (EVA-CLIP-18B)
