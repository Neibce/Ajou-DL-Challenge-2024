# Ajou-DL-Challenge-2024

- SW중심대학연합 제3회 아주 소중한 딥러닝 챌린지 (24.07.26. ~ 24.08.30.)
- Public, Private score 1st
- Zero-shot classification

![private](https://github.com/user-attachments/assets/0fc8d319-017d-42a2-821d-f7bcf985f24a)

## Reference
### [Demystifying CLIP Data](https://arxiv.org/abs/2309.16671) (2023)
Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer and Christoph Feichtenhofer

### [EVA-CLIP-18B: Scaling CLIP to 18 Billion Parameters](https://arxiv.org/abs/2402.04252) (2023)
Quan Sun, Jinsheng Wang, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang and Xinlong Wang

### GitHub Repositories: [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip), [huggingface/transformers](https://github.com/huggingface/transformers), [facebookresearch/MetaCLIP](https://github.com/facebookresearch/MetaCLIP), [baaivision/EVA](https://github.com/baaivision/EVA/tree/master/EVA-CLIP-18B)

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
dataset의 이미지들이 블러 처리가 된 이미지인 것을 확인, [RandomAdjustSharpness](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomAdjustSharpness.html)를 적용해 보았고, 이를 통해 약 1%의 성능 향상(public 기준)이 있음을 알 수 있었음.
[`torchvision.transforms.functional.adjust_sharpness`](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.adjust_sharpness.html)으로도 대체 가능할 것으로 보임.

## Prompt-Tuning
- 주요 프롬프트
```python
    prompts.append(f"a blurry photo of a {class_name}")
    prompts.append(f"a blurry image of a {class_name}")
    prompts.append(f"a blurred photo of a {class_name}")
    prompts.append(f"a blurred image of a {class_name}")
```
ChatGPT 서비스에 dataset의 이미지 10장을 임의로 골라 설명을 시켰을 때, blur라는 단어가 공통적으로 들어감을 확인. 이에 착안해 위의 프롬프트들을 사용하였고 약 1%(public 기준)의 성능 향상을 가져올 수 있었음.

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

### [EVA-CLIP-18B](https://github.com/baaivision/EVA/tree/master/EVA-CLIP-18B) (EVA-CLIP-18B)
```python
model = AutoModel.from_pretrained(
    'BAAI/EVA-CLIP-18B',
    torch_dtype=torch.float16,
    trust_remote_code=True).to('cuda').eval()
```
## DataLoader
### For MetaCLIP
```python
ds_meta = ImageFolder(os.path.join(root, dataset_name), transform=processor_23)
ds_meta.samples = natsorted(ds_meta.samples)
dl_meta = DataLoader(ds_meta, shuffle=False, batch_size=32, num_workers=2)
```
### For EVA-CLIP
```python
ds_eva = ImageFolder(os.path.join(root, dataset_name), transform=processor_20)
ds_eva.samples = natsorted(ds_eva.samples)
dl_eva = DataLoader(ds_eva, shuffle=False, batch_size=32, num_workers=2)
```

## Zero-shot Classification
### For MetaCLIP
```python
meta_probs_list = []

with torch.no_grad(), torch.cuda.amp.autocast():
    text = tokenizer.tokenize(prompts).to(device)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    for x, y in tqdm(dl_meta):
        x = x.to(device)
        image_features = model.encode_image(x)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        zero_shot_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        meta_probs_list += zero_shot_probs
```
### For EVA-CLIP
```python
eva_probs_list = []

with torch.no_grad(), torch.cuda.amp.autocast():
    text = tokenizer(prompts, return_tensors='pt', padding=True).input_ids.to('cuda')
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    for x, y in tqdm(dl_eva):
        x = x.to(device)
        image_features = model.encode_image(x)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        zero_shot_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        eva_probs_list += zero_shot_probs
```

## Enassemble
### MetaCLIP * 0.5 + EVA-CLIP * 0.5
```python
ensembled_probs_list = [meta_probs * 0.5 + eva_probs * 0.5 for meta_probs, eva_probs in zip(meta_probs_list, eva_probs_list)]
label_list = [ensembled_probs.reshape(len(class_names), -1).mean(dim=-1).max(dim=-1)[1].tolist() for ensembled_probs in ensembled_probs_list]
```

## 결론
