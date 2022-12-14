from PIL import Image
import os
import numpy as np
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel
import torch
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.transforms as transforms

# Load up pretrained CLIP model
device = "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model.to(device)

data_dir = "../MapCap_partition"
classes = os.listdir(data_dir) # assumes data_dir/label/image structure

class_performance = defaultdict(list)
class_maps = defaultdict(list)

transform = transforms.Compose([
    transforms.PILToTensor()
])

for cl in classes:
    
    print(cl)
    class_index = classes.index(cl)
    class_path = os.path.join(data_dir, cl)
    im_paths = os.listdir(class_path)

    ims = [transform(Image.open(os.path.join(class_path, im_path)).convert('RGB')).to(device) for im_path in im_paths]

    # Run inference for each image
    for image_idx in range(len(ims)):
        image = ims[image_idx]
        inputs = processor(text=classes, images=image, return_tensors="pt", padding=True)

        outputs = model(**inputs)

        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

        prediction = probs.argmax()

        if prediction == class_index:
            class_performance[cl].append(1)
        else:
            class_performance[cl].append(0)


acc_df = {}
df_classes = []
df_accs = []
for cl, acc in class_performance.items():
    df_classes.append(cl)
    df_accs.append(np.mean(acc))

acc_df["Class"] = df_classes
acc_df["Accuracy"] = df_accs

pd.DataFrame.from_dict(acc_df).to_csv("CLIP_Zero_Shot_Accuracy.csv")
