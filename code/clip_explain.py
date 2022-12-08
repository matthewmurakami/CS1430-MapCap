import torch
import clip.clip as clip
import numpy as np
import os, shutil
import torchvision
from skimage import segmentation
import matplotlib.pyplot as plt

# Number of layers for image Transformer
start_layer =  -1

# Number of layers for text Transformer
start_layer_text =  -1

def interpret(image, texts, model, device, start_layer=start_layer, start_layer_text=start_layer_text):
    batch_size = image.shape[0]
    # images = image.repeat(batch_size, 1, 1, 1)
    logits_per_image, logits_per_text = model(image, texts)

    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.to(device) * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    if start_layer == -1: 
      # calculate index of last layer 
      start_layer = len(image_attn_blocks) - 1
    
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]

    
    text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

    if start_layer_text == -1: 
      # calculate index of last layer 
      start_layer_text = len(text_attn_blocks) - 1

    num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
    R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
    R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(text_attn_blocks):
        if i < start_layer_text:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R_text = R_text + torch.bmm(cam, R_text)
    text_relevance = R_text
   
    return text_relevance, image_relevance


# First attempt at segmenting the heatmap
def segment_heatmap(heatmap):
    dim = int(heatmap.numel() ** 0.5)
    heatmap = heatmap.reshape(1, 1, dim, dim)
    heatmap = torch.nn.functional.interpolate(heatmap, size=224, mode='bilinear')
    heatmap = heatmap.reshape(1, 224, 224).to(device).data.cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    segments = []

    threshold = .4
    mask = np.where(heatmap >= threshold, 2, 0)
    while True:
        one_elements = np.argwhere(mask == 2)
        seed_pixel = one_elements[0] if len(one_elements) != 0 else []

        if len(seed_pixel) == 0:
            break

        mask = segmentation.flood_fill(mask, tuple(seed_pixel), 1)
        curr_mask = mask == 1
        segments.append((heatmap * curr_mask, threshold))

        mask = mask - (2 * curr_mask) # Get rid of segment you just extracted

    return segments


def gather_data():
    # Reads in the Map Cap partition classes
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(224), torchvision.transforms.ToTensor()])

    dataset = torchvision.datasets.ImageFolder("../MapCap_partition", transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    return dataloader
    

if torch.cuda.is_available():
    device = "cuda"
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
device = "cpu"

model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

data = gather_data()
newpath = "../MapCap_segments"
if os.path.exists(newpath):
    shutil.rmtree(newpath)
os.makedirs(newpath)

imgnum = 0
for image, label in data:
    texts = [""]
    text = clip.tokenize(texts).to(device)

    R_text, R_image = interpret(model=model, image=image, texts=text, device=device)
    batch_size = text.shape[0]

    for i in range(batch_size):
        # From here on, we're trying to segment the heatmap into distinct patches
        segments = segment_heatmap(R_image[i])
        segnum = 0
        for segment in segments:
            segmented_image = torch.tensor(segment[0]) * image.squeeze(0)
            classPath = os.path.join(newpath, str(label.item()))
            if not os.path.exists(classPath):
                os.makedirs(classPath)
            torchvision.utils.save_image(segmented_image,  os.path.join(classPath, "img%i-seg%i.JPEG"%(imgnum,segnum)))
            segnum += 1
    imgnum += 1



    

