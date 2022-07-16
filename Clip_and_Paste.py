# %%
# !! {"metadata":{
# !!   "id":"cc-imports"
# !! }}

#<cc-imports>

import subprocess
import os

# %%
# !! {"metadata":{
# !!   "id": "view-in-github",
# !!   "colab_type": "text"
# !! }}
"""
<a href="https://colab.research.google.com/github/robgon-art/CLIPandPASTE/blob/main/CLIP_and_PASTE.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

# %%
# !! {"metadata":{
# !!   "id": "5XEbqN5n8RQx"
# !! }}
"""
# **CLIP and PASTE: Using AI to Create Photo Collages from Text Prompts**
## How to use ML models to extract objects from photographs and rearrange them to create modern art

![CLIPandPadte Cover Image](https://raw.githubusercontent.com/robgon-art/CLIPandPASTE/main/cover%20shot%20mid.jpg)

**By Robert. A Gonsalves**</br>

You can see my article on Medium.

The source code and generated images are released under the [CC BY-SA license](https://creativecommons.org/licenses/by-sa/4.0/).</br>
![CC BYC-SA](https://licensebuttons.net/l/by-sa/3.0/88x31.png)

## Google Colabs
* [CLIP and PASTE](https://colab.research.google.com/github/robgon-art/CLIPandPASTE/blob/main/CLIP_and_PASTE.ipynb)

## Acknowledgements
- CLIP by A. Radford et al., Learning Transferable Visual Models From Natural Language Supervision (2021)
- F. Boudin, PKE: An Open-Source Python-based Keyphrase Extraction Toolkit (2016), Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: System Demonstrations
- Wikimedia Commons (2004-present)
- OpenImages (2020)
- GRoIE by L. Rossi, A. Karimi, and A. Prati, A Novel Region of Interest Extraction Layer for Instance Segmentation (2020)
- D. P. Kingma and J. Lei Ba, Adam: A Method for Stochastic Optimization (2015), The International Conference on Learning Representations 2015
- M. Grootendorst, KeyBERT: Minimal keyword extraction with BERT (2020)
- E. Riba, D. Mishkin, D. Ponsa, E. Rublee, and G. Bradski, Kornia: an Open Source Differentiable Computer Vision Library for PyTorch (2020), Winter Conference on Applications of Computer Vision

## Citation
To cite this repository:

```bibtex
@software{CLIP and PASTE,
  author  = {Gonsalves, Robert A.},
  title   = {CLIP and PASTE: Using AI to Create Photo Collages from Text Prompts},
  url     = {https://github.com/robgon-art/CLIPandPAST},
  year    = 2022,
  month   = June
}
```

"""

# %%
# !! {"metadata":{
# !!   "id": "7ysKe2XpxRFd",
# !!   "cellView": "form"
# !! }}
#@title Initialize the System
sub_p_res = subprocess.run(['pip', 'install', 'kornia'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', 'ftfy'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', 'git+https://github.com/openai/CLIP.git', '--no-deps'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', 'boto3', '>', '/dev/null', '2>&1'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['gdown', '--id', '1TS5K0BGk5ruCF-bc6yeMSAEb5z8Oi_st'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['gdown', '--id', '1-2ForMsp58l6DVAeUqEvW0N24-YITf5o'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['wget', 'https://raw.githubusercontent.com/openimages/dataset/master/downloader.py'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', 'git+https://github.com/boudinfl/pke.git'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', 'mmcv-full', '-f', 'https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['rm', '-rf', 'mmdetection'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['git', 'clone', 'https://github.com/open-mmlab/mmdetection.git'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
os.chdir('mmdetection') #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', '-e', '.'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['mkdir', 'checkpoints'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['wget', '-c', 'https://download.openmmlab.com/mmdetection/v2.0/groie/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco_20200604_211715-42eb79e1.pth', '\\'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
    #-O checkpoints/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco_20200604_211715-42eb79e1.pth
os.chdir('/media/ws-ml/Data-ml1/ML_Images/CLIPandPASTE/') #<cc-cm>
"""
sub_p_res = subprocess.run(['rm', '-r', 'open_images'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['rm', '-r', 'wiki_images'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['mkdir', 'open_images'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['mkdir', 'wiki_images'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
"""
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import kornia
import numpy as np
import pickle
import requests
import shutil
import os
import cv2
import re
import clip
import torch
import warnings
from PIL import Image
import matplotlib.pyplot as plt
import json
import random
import nltk
import pke
import matplotlib.pylab as plb
import torchvision.transforms as T
# from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import torchvision
import sys
sys.path.append("/media/ws-ml/Data-ml1/ML_Images/CLIPandPASTE/mmdetection")
import mmdet
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
import IPython
from shapely.geometry import Polygon
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

config = '/media/ws-ml/Data-ml1/ML_Images/CLIPandPASTE/mmdetection/configs/groie/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco.py'
checkpoint = '/media/ws-ml/Data-ml1/ML_Images/CLIPandPASTE/mmdetection/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco_20200604_211715-42eb79e1.pth'
groie_model = init_detector(config, checkpoint, device='cuda:0')

text_features16 = np.load("ai-memer_embeddings16.npy")
print(text_features16.shape)
annotations = pickle.load(open("ai-memer_annotations.pkl", "rb"))
print(annotations[520000])

device = torch.device('cuda')
clip_model, clip_preprocess = clip.load('ViT-B/32', device, jit=False)

def get_text_features(prompt):
  text_input = clip.tokenize(prompt).to(device)
  with torch.no_grad():
    text_features = clip_model.encode_text(text_input)
  text_features /= text_features.norm(dim=-1, keepdim=True)
  return(text_features)

def get_top_N_semantic_similarity(similarity_list, N):
  results = zip(range(len(similarity_list)), similarity_list)
  results = sorted(results, key=lambda x: x[1],reverse = True)
  scores = []
  indices = []
  for index,score in results[:N]:
    scores.append(score)
    indices.append(index)
  return scores, indices

extractor = pke.unsupervised.TopicRank()
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
print(stop_words)

def download_file(url, path):
  filename = url.split("/")[-1]
  file_path = os.path.join(path, filename)
  headers = {'User-Agent': 'CLIPandPASTE/1.0 (https://robgon.medium.com/; robgon.art@gmail.com)'}
  response = requests.get(url, headers=headers)
  file = open(file_path, "wb")
  file.write(response.content)
  file.close()

coco_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
        'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
        'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']
  
colors = [[random.randint(0, 255) for _ in range(3)] for _ in coco_names]
img_size = 224

# from https://stackoverflow.com/questions/61616810/how-to-do-cubic-spline-interpolation-and-integration-in-pytorch

def h_poly_helper(tt):
  A = torch.tensor([
      [1,  0, -3,  2],
      [0,  1, -2,  1],
      [0,  0,  3, -2],
      [0,  0, -1,  1]
      ], dtype=tt[-1].dtype)
  return [
    sum( A[i, j]*tt[j] for j in range(4) )
    for i in range(4) ]

def h_poly(t):
  tt = [ None for _ in range(4) ]
  tt[0] = 1
  for i in range(1, 4):
    tt[i] = tt[i-1]*t
  return h_poly_helper(tt)

def interp(x, y, xs):
  m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
  m = torch.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
  I = plb.searchsorted(x[1:], xs)
  dx = (x[I+1]-x[I])
  hh = h_poly((xs-x[I])/dx)
  return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx

def calculate_iou(box1, box2):
    poly1 = Polygon([[box1[0], box1[1]], [box1[2], box1[1]], [box1[2], box1[3]], [box1[0], box1[3]]])
    poly2 = Polygon([[box2[0], box2[1]], [box2[2], box2[1]], [box2[2], box2[3]], [box2[0], box2[3]]])
    iou = poly1.intersection(poly2).area / poly1.union(poly2).area
    return iou

# %%
# !! {"metadata":{
# !!   "id": "3lhlBo_vmJHk",
# !!   "cellView": "form",
# !!   "outputId": "0324805d-8e13-4fdd-e8e4-8a2114b4058b",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 1000
# !!   }
# !! }}
#@title Find Source Images

all_images = []
all_files = []


prompt = "penguins skiing down a snowy mountain" #@param {type:"string"}
prompt = prompt.lower()
num_keywords = 10
image_size = 256

extractor.load_document(input=prompt, language='en')
extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ', 'VERB'})
extractor.candidate_weighting()

keyphrases = extractor.get_n_best(n=num_keywords, stemming=False)
print(keyphrases)
keywords = [prompt]
for i, (candidate, score) in enumerate(keyphrases):   
  print("rank {}: {} ({})".format(i, candidate, score))
  if candidate not in keywords:
    keywords.append(candidate)

words = prompt.split()
for w in words:
  if w not in keywords and w not in stop_words:
    keywords.append(w)

keywords = keywords[:10]
print(keywords)

warnings.filterwarnings('ignore')

all_images = []
all_files = []
total_images = 32
num_openimages = 0
num_wikiimages = 0

num_images = int(round(float(total_images)/(len(keywords)+1)+0.5))

for j in range(len(keywords)):
  print("looking for" , num_images*2, "images of", keywords[j])
  feature_text = "<|startoftext|> Image of " + keywords[j] + " <|endoftext|>"
  query_features = get_text_features(feature_text)
  text_similarity = query_features.cpu().numpy() @ text_features16.T
  text_similarity = text_similarity[0]
  text_scores, text_indices = get_top_N_semantic_similarity(text_similarity, N=num_images)

  # get the images from OpenImages
  # print("Downloading images from OpenImages")
  f = open("images.txt", "w")
  for i in text_indices:
    f.write(annotations[i][0] +"\n")
  f.close()
  sub_p_res = subprocess.run(['python', 'downloader.py', 'images.txt', '--download_folder=open_images', '--num_processes=1', '>', '/dev/null', '2>&1'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
  print(sub_p_res) #<cc-cm>
  for i in range(0, num_images):
    image_id = annotations[text_indices[i]][0]
    parts = image_id.split("/")
    file_path = "open_images/" + parts[1] + ".jpg"
    if file_path not in all_files:
      # print(file_path)
      img = Image.open(file_path)
      img = img.convert(mode="RGB")
      all_images.append(img)
      all_files.append(file_path)
      num_openimages += 1
  """
  # get the images Wikipedia
  s = requests.Session()
  url = "https://commons.wikimedia.org/w/api.php"
  params = {
      "action": "query",
      "generator": "images",
      "prop": "imageinfo",
      "gimlimit": 500,
      "titles": keywords[j],
      "iiprop": "url|dimensions",
      "format": "json"
  }
  r = s.get(url=url, params=params)
  data = r.json()
  image_files = []
  if "query" not in data.keys():
    continue
  pages = data['query']['pages']
  for k, v in pages.items():
    for info in v['imageinfo']:
      imurl = info["url"]
      h =  info["height"]
      w = info["width"]
      a = h * w
      if a >= 512*512 and imurl not in image_files and imurl.lower().endswith("jpg"):
        image_files.append(imurl)
  random.shuffle(image_files)
  for im in image_files[:num_images]:
    filename = im.split("/")[-1]
    download_file(im, "wiki_images")
    file_path = "wiki_images/" + filename
    if file_path not in all_files:
      # print(file_path)
      img = Image.open(file_path)
      img = img.convert(mode="RGB")
      all_images.append(img)
      all_files.append(file_path)
      num_wikiimages += 1

print("num openimages  ", num_openimages)
print("num wiki images ", num_wikiimages)
print("num total images", num_openimages+num_wikiimages)
"""

# End Wiki Media and Open Images

input_resolution = 224
image_features = torch.empty((0, image_size))

preprocess = Compose([
    Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
    CenterCrop(input_resolution),
    ToTensor()
])

# Added this code for using custom images
import glob
file_path_list = glob.glob("./content/custom_images/*.jpg")
for file_path in file_path_list:
      img = Image.open(file_path)
      img = img.convert(mode="RGB")
      all_images.append(img)
      all_files.append(file_path)
print("num total images", len(all_images))

# Added this code for using custom images

images = [preprocess(im) for im in all_images]
image_input = torch.tensor(np.stack(images)).cuda()
with torch.no_grad():
  image_features = clip_model.encode_image(image_input).float().cpu()  
image_features /= image_features.norm(dim=-1, keepdim=True)

feature_text = "<|startoftext|> Image of " + prompt + " <|endoftext|>"
query_features = get_text_features(feature_text)

image_similarity = query_features.cpu().numpy() @ image_features.numpy().T
image_similarity = image_similarity[0]
print(len(all_files))

num_images = min(int(0.75*len(all_files)), 25)
image_scores, image_indices = get_top_N_semantic_similarity(image_similarity, N=num_images)
columns = 5
rows = num_images // columns + 1
fig=plt.figure(figsize=(columns*5, rows*5))
for i in range(1, columns*rows + 1):
  file_name = all_files[image_indices[i-1]]
  img = Image.open(file_name)
  img = img.convert(mode="RGB")
  fig.add_subplot(rows, columns, i)
  plt.margins(y=10)
  plt.imshow(img)
  plt.text(0, -30, str(i) + " " + file_name, fontsize=10)
  plt.axis("off")
  if i >= num_images:
    break
plt.show()

image_parts = []
parts_rgb = []
parts_a = []
part_sizes = {}
part_count = 0

preprocess_parts = Compose([
    ToTensor()
])

for i in range(num_images):
  image_file = all_files[image_indices[i]]
  print(i, file_name)

  input_image = Image.open(image_file).convert(mode="RGB")
  plt.figure(figsize=(10, 10))
  plt.axis("off")
  _ = plt.imshow(input_image)
  plt.show()

  result = inference_detector(groie_model, image_file)
  bbox_result, segm_result = result

  boxes = []
  overlaps = []
  scores = []
  labels = []
  mask_areas = []
  result_image = np.array(input_image.copy())
  count = 0

  # print()
  # print("objects")

  for label, boxscores in enumerate(bbox_result):
    for boxscore in boxscores:
      box = boxscore[:4]
      score = boxscore[4]

      overlapping = False
      for b in boxes:
        overlap = calculate_iou(box, b)
        # print("overlap", overlap)
        if  overlap > 0.85:
          overlapping = True
          # print("skipping")
          break

      overlaps.append(overlapping)

      boxes.append(box)
      scores.append(score)
      labels.append(label)
      # print(label, coco_names[label], box, score)

      if overlapping:
        continue

      color = random.choice(colors)
      # print(count+1, coco_names[label], round(100*score.item(), 2))
      # draw box
      tl = round(0.001 * max(result_image.shape[0:2])) + 1  # line thickness
      c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
      cv2.rectangle(result_image, c1, c2, color, thickness=tl)
      # draw text
      display_txt = "%s: %.1f%%" % (coco_names[label], 100*score)
      tf = max(tl - 1, 1)  # font thickness
      t_size = cv2.getTextSize(display_txt, 0, fontScale=tl / 3, thickness=tf)[0]
      c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
      cv2.rectangle(result_image, c1, c2, color, -1)  # filled
      cv2.putText(result_image, display_txt, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
      count += 1

  if count == 0:
    print("no objects found")
    continue

  plt.figure(figsize=(10, 10))
  plt.axis("off")
  _ = plt.imshow(result_image)
  plt.show()

  count = 0
  masks = []
  mask_accum = None
  for object_masks in segm_result:
    for mask in object_masks:
      mask_np = np.float32(mask)
      masks.append(mask_np)
      mask_area = mask_np.sum() / (input_image.width*input_image.height)
      mask_areas.append(mask_area)
      if mask_accum is None:
        mask_accum = mask
      else:
        mask_accum = np.maximum(mask_accum, mask)
      count += 1

  images = []

  img_wid = input_image.width
  img_hgt = input_image.height

  count = 0
  for box, overlapping, score, mask_np in zip(boxes, overlaps, scores, masks):
    # print(box)

    mask_np = np.expand_dims(mask_np, axis=2)
    num_mask_pixels = mask_np.sum()

    if overlapping or num_mask_pixels < 1000:
      continue

    mask_np = np.repeat(mask_np, 3, axis=2)
    image_np = np.array(input_image, dtype=np.float32)/255.0

    if mask_np.shape != image_np.shape:
      continue

    masked_image_np = mask_np * image_np

    box_lft = int(box[0].item())
    box_top = int(box[1].item())
    box_rgt = int(box[2].item())
    box_bot = int(box[3].item())

    cutout_image_np = masked_image_np[box_top:box_bot, box_lft:box_rgt]
    cutout_mask_np = mask_np[box_top:box_bot, box_lft:box_rgt]

    box_wid = box_rgt - box_lft
    box_hgt = box_bot - box_top

    if box_wid > box_hgt: # handle landscape images
      # print("landscape")
      pad = (box_wid - box_hgt) // 2
      padded_image_np = np.zeros((box_wid, box_wid, 3), dtype=cutout_image_np.dtype)
      padded_image_np[pad:pad+box_hgt, :] = cutout_image_np
    
    else: # handle portrait images
      # print("portrait")
      pad = (box_hgt - box_wid) // 2
      padded_image_np = np.zeros((box_hgt, box_hgt, 3), dtype=cutout_image_np.dtype)
      padded_image_np[:, pad:pad+box_wid] = cutout_image_np     

    image_PIL = Image.fromarray(np.uint8(padded_image_np*255))

    plt.figure(figsize=(6, 6))
    plt.axis("off")
    _ = plt.imshow(image_PIL)
    plt.show()

    image_parts.append(preprocess(image_PIL))

    part_PIL = Image.fromarray(np.uint8(cutout_image_np*255))
    w, h = part_PIL.size
    if w > h and w > image_size:
      part_pil = part_PIL.resize((image_size, int(h*image_size/w)), Image.BICUBIC)
    elif h > image_size:
      part_pil = part_PIL.resize((int(w*image_size/h), image_size), Image.BICUBIC)

    parts_rgb.append(preprocess_parts(part_PIL))
    mask_PIL = Image.fromarray(np.uint8(cutout_mask_np*255))
    parts_a.append(preprocess_parts(mask_PIL))
    part_sizes[part_count] = num_mask_pixels
    count += 1
    part_count += 1

to_pil = T.ToPILImage()

num_parts = min(len(image_parts)//2,100)

part_input = torch.tensor(np.stack(image_parts)).cuda()
with torch.no_grad():
  part_features = clip_model.encode_image(part_input).float().cpu()  
part_features /= part_features.norm(dim=-1, keepdim=True)

feature_text = "<|startoftext|> Image of " + prompt + " <|endoftext|>"
query_features = get_text_features(feature_text)

part_similarity = query_features.cpu().numpy() @ part_features.numpy().T
part_similarity = part_similarity[0]

part_scores, part_indices = get_top_N_semantic_similarity(part_similarity, N=num_parts)
columns = 5
rows = num_parts // columns + 1
fig=plt.figure(figsize=(columns*5, rows*5))
for i in range(1, columns*rows + 1):
  img = to_pil(image_parts[part_indices[i-1]])
  fig.add_subplot(rows, columns, i)
  plt.margins(y=10)
  plt.imshow(img)
  plt.text(0, -5, str(i-1), fontsize=12)
  plt.axis("off")
  if i >= num_parts:
    break
plt.show()

ordered_part_indices = []
for p in part_indices[:num_parts]:
  size = part_sizes[p]
  ordered_part_indices.append((size, p))

ordered_part_indices.sort(reverse=True)

# %%
# !! {"metadata":{
# !!   "id": "jYv7Wb0n8oU5"
# !! }}
"""
### Create Layouts
"""

# %%
# !! {"metadata":{
# !!   "id": "2uTvRf583fPX",
# !!   "outputId": "c4c87cb2-92bb-411f-cc0f-22a8e5a5de1a",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 352,
# !!     "referenced_widgets": [
# !!       "12dfc8f3be7043568f211ebd847d1ba4",
# !!       "bbf8ce574f9a40d8b0e31cc374893927",
# !!       "2029bd4e74f242328b24e604fb3832c1",
# !!       "bd0d78d3d49b422fb5a2520d3e79469d",
# !!       "c9e18976325f48c295a1e1b5fa09fef7",
# !!       "430774531dee4cf99010397bb5f445e6",
# !!       "89a4e4d1e0ad419aafe26916747ea69c",
# !!       "7b5e46be957d48779549bc684cb73cae",
# !!       "9ab76575971f493a98af39cad78244c1",
# !!       "a96fea00e61a4a0492d5efe37da99804",
# !!       "ccaf4675d9f344be8dc4c80baccf9e2e"
# !!     ]
# !!   }
# !! }}
num_initial_layouts = 100
num_ctrl_ponts = 5
init_rand_amount = 0.25
num_shapes = 30

resize_factor = 1.0
new_img_size = int(img_size*resize_factor)

bg_x = torch.linspace(0, img_size-1, num_ctrl_ponts).to(device)
bgvals = (0.5 + init_rand_amount/2.0 * torch.rand(size=(3, num_ctrl_ponts))).to(device) 
bgvals.requires_grad = True
bg_xs = torch.linspace(0, img_size-1, img_size).to(device)

img_0 = interp(bg_x.cpu(), bgvals[0].cpu(), bg_xs.cpu()).to(device)
img_1 = interp(bg_x.cpu(), bgvals[1].cpu(), bg_xs.cpu()).to(device)
img_2 = interp(bg_x.cpu(), bgvals[2].cpu(), bg_xs.cpu()).to(device)
img = torch.vstack([img_0, img_1, img_2])

img = img.permute(1,0)
img = img.tile((img_size, 1, 1))
img = img.unsqueeze(0)
img = img.permute(0, 3, 2, 1) # NHWC -> NCHW
img = torch.nn.functional.interpolate(img, scale_factor=resize_factor, mode="bilinear")

bg_img = img.clone()

image_list = []
param_list = []

from tqdm import tqdm_notebook as tqdm

print("Creating", num_initial_layouts, "layouts for analysis")

for j in tqdm(range(num_initial_layouts)):
  img = bg_img.clone()
  partvals = torch.zeros(size=(num_shapes, 2)).to(device)

  for i, index in enumerate(ordered_part_indices[:num_shapes]):
    # get the part
    part = parts_rgb[index[1]].to(device)
    mask = parts_a[index[1]].to(device)

    # scale
    scale_factor = torch.tensor([new_img_size/2000.0, new_img_size/2000.0]).to(device)
    part = kornia.geometry.transform.scale(part[None, :], scale_factor[None, :]).squeeze()
    mask = kornia.geometry.transform.scale(mask[None, :], scale_factor[None, :]).squeeze()

    # pad
    h = part.shape[1]
    w = part.shape[2]
    lft_pad = (new_img_size - w)//2
    top_pad = (new_img_size - h)//2
    rgt_pad = new_img_size - w - lft_pad
    bot_pad = new_img_size - h - top_pad
    part = T.functional.pad(part, (lft_pad, top_pad, rgt_pad, bot_pad))
    mask = T.functional.pad(mask, (lft_pad, top_pad, rgt_pad, bot_pad))

    # translate
    w_range = new_img_size - w * new_img_size/2000
    h_range = new_img_size - h * new_img_size/2000
    partvals[i][1] = (random.random()-0.5) * h_range / new_img_size
    partvals[i][0] = (random.random()-0.5) * w_range / new_img_size
    trans = partvals[i] * new_img_size
    part = kornia.geometry.transform.translate(part[None, :], trans[None, :]).squeeze()
    mask = kornia.geometry.transform.translate(mask[None, :], trans[None, :]).squeeze()

    # composite the part
    img *= 1-mask
    img += part

  image_list.append(img)
  param_list.append(partvals.clone())

layout_input = torch.stack(image_list).cuda().squeeze()
with torch.no_grad():
  layout_features = clip_model.encode_image(layout_input).float().cpu() 
layout_features /= layout_features.norm(dim=-1, keepdim=True)

layout_similarity = query_features.cpu().numpy() @ layout_features.numpy().T
layout_similarity = layout_similarity[0]
layout_scores, layout_indices = get_top_N_semantic_similarity(layout_similarity, N=num_parts)

img = image_list[layout_indices[0]]
image = img.detach().cpu().numpy()
image = np.transpose(image, (0, 2, 3, 1))[0]
image = np.clip(image*255, 0, 255).astype(np.uint8)
image_pil = Image.fromarray(image)
plt.figure(figsize=(5, 5))
img = plt.imshow(image_pil)
plt.axis('off')
plt.show()

# %%
# !! {"metadata":{
# !!   "id": "kfCX25-I8wic"
# !! }}
"""
### Optimize Layouts
"""

# %%
# !! {"metadata":{
# !!   "id": "x056HMOtFyfc",
# !!   "outputId": "b0d25da1-4fed-4a44-d845-4cd700ab9822",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 1000
# !!   }
# !! }}
num_steps = 100
color_lr = 0.005
parts_lr = 0.01
num_augmentations = 32

text_features = get_text_features(prompt)

augment_trans = T.Compose([
  T.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
  T.RandomResizedCrop(img_size, scale=(0.7,0.9)),
  T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
  T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

bg_x = torch.linspace(0, img_size-1, num_ctrl_ponts).to(device)
bgvals = (0.5 + init_rand_amount/2.0 * torch.rand(size=(3, num_ctrl_ponts))).to(device) 
bgvals.requires_grad = True
bg_xs = torch.linspace(0, img_size-1, img_size).to(device)
# print(bgvals)

partvals = param_list[layout_indices[0]].clone()
partvals.requires_grad = True

bg_optim = torch.optim.Adam([{'params': bgvals,   'lr': color_lr},
                             {'params': partvals, 'lr': parts_lr}])
loss_fn = torch.nn.CosineEmbeddingLoss()
target = torch.full((1,32), fill_value=1.0).squeeze().to(device)

# Run the main optimization loop
for t in range(num_steps+1):
  bg_optim.zero_grad()

  img_0 = interp(bg_x.cpu(), bgvals[0].cpu(), bg_xs.cpu()).to(device)
  img_1 = interp(bg_x.cpu(), bgvals[1].cpu(), bg_xs.cpu()).to(device)
  img_2 = interp(bg_x.cpu(), bgvals[2].cpu(), bg_xs.cpu()).to(device)
  img = torch.vstack([img_0, img_1, img_2])

  img = img.permute(1,0)
  img = img.tile((img_size, 1, 1))
  img = img.unsqueeze(0)
  img = img.permute(0, 3, 2, 1) # NHWC -> NCHW

  for index, params in zip(ordered_part_indices[:num_shapes], partvals):
    # get the part
    part = parts_rgb[index[1]].to(device)
    mask = parts_a[index[1]].to(device)

    # scale
    scale_factor = torch.tensor([img_size/2000.0, img_size/2000.0]).to(device)
    part = kornia.geometry.transform.scale(part[None, :], scale_factor[None, :]).squeeze()
    mask = kornia.geometry.transform.scale(mask[None, :], scale_factor[None, :]).squeeze()

    # pad
    h = part.shape[1]
    w = part.shape[2]
    lft_pad = (img_size - w)//2
    top_pad = (img_size - h)//2
    rgt_pad = img_size - w - lft_pad
    bot_pad = img_size - h - top_pad
    part = T.functional.pad(part, (lft_pad, top_pad, rgt_pad, bot_pad))
    mask = T.functional.pad(mask, (lft_pad, top_pad, rgt_pad, bot_pad))

    # translate
    # trans = (params-0.5) * 150
    trans = params * img_size
    part = kornia.geometry.transform.translate(part[None, :], trans[None, :]).squeeze()
    mask = kornia.geometry.transform.translate(mask[None, :], trans[None, :]).squeeze()

    # composite the part
    img *= 1-mask
    img += part

  img_augs = []
  for n in range(num_augmentations):
    img_augs.append(augment_trans(img))
  im_batch = torch.cat(img_augs)
  image_features = clip_model.encode_image(im_batch)
  loss = loss_fn(image_features, text_features, target)

  loss.backward()
  bg_optim.step()
  if t % 10 == 0:
    print("-" * 10)
    image = img.detach().cpu().numpy()
    image = np.transpose(image, (0, 2, 3, 1))[0]
    image = np.clip(image*255, 0, 255).astype(np.uint8)
    image_pil = Image.fromarray(image)
    print('render loss:', loss.item())
    print('iteration:', t)
    plt.figure(figsize=(5, 5))
    img = plt.imshow(image_pil)
    plt.axis('off')
    plt.show()


# %%
# !! {"metadata":{
# !!   "id": "clcqdd3p83J-"
# !! }}
"""
### Display High Resolution Image
"""

# %%
# !! {"metadata":{
# !!   "id": "LYYyGEp4wzSG",
# !!   "outputId": "6fcb2a6e-8d29-4f01-c7f8-1823d2fbc086",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 913
# !!   }
# !! }}
resize_factor = 4.0
new_img_size = int(img_size*resize_factor)

img_0 = interp(bg_x.cpu(), bgvals[0].cpu(), bg_xs.cpu()).to(device)
img_1 = interp(bg_x.cpu(), bgvals[1].cpu(), bg_xs.cpu()).to(device)
img_2 = interp(bg_x.cpu(), bgvals[2].cpu(), bg_xs.cpu()).to(device)
img = torch.vstack([img_0, img_1, img_2])

img = img.permute(1,0)
img = img.tile((img_size, 1, 1))
img = img.unsqueeze(0)
img = img.permute(0, 3, 2, 1) # NHWC -> NCHW
img = torch.nn.functional.interpolate(img, scale_factor=resize_factor, mode="bilinear")

for index, params in zip(ordered_part_indices, partvals):
  # get the part
  part = parts_rgb[index[1]].to(device)
  mask = parts_a[index[1]].to(device)

  # scale
  scale_factor = torch.tensor([new_img_size/2000.0, new_img_size/2000.0]).to(device)
  part = kornia.geometry.transform.scale(part[None, :], scale_factor[None, :]).squeeze()
  mask = kornia.geometry.transform.scale(mask[None, :], scale_factor[None, :]).squeeze()

  # pad
  h = part.shape[1]
  w = part.shape[2]
  lft_pad = (new_img_size - w)//2
  top_pad = (new_img_size - h)//2
  rgt_pad = new_img_size - w - lft_pad
  bot_pad = new_img_size - h - top_pad
  part = T.functional.pad(part, (lft_pad, top_pad, rgt_pad, bot_pad))
  mask = T.functional.pad(mask, (lft_pad, top_pad, rgt_pad, bot_pad))

  # translate
  # trans = (params-0.5) * 150 * resize_factor
  trans = params * new_img_size
  part = kornia.geometry.transform.translate(part[None, :], trans[None, :]).squeeze()
  mask = kornia.geometry.transform.translate(mask[None, :], trans[None, :]).squeeze()

  # composite the part
  img *= 1-mask
  img += part

image = img.detach().cpu().numpy()
image = np.transpose(image, (0, 2, 3, 1))[0]
image = np.clip(image*255, 0, 255).astype(np.uint8)
image_pil = Image.fromarray(image)
image_pil.save("out.png")

import IPython
IPython.display.Image("out.png")

# %%
# !! {"main_metadata":{
# !!   "accelerator": "GPU",
# !!   "colab": {
# !!     "collapsed_sections": [],
# !!     "name": "CLIP and PASTE",
# !!     "provenance": [],
# !!     "include_colab_link": true
# !!   },
# !!   "kernelspec": {
# !!     "display_name": "Python 3",
# !!     "name": "python3"
# !!   },
# !!   "language_info": {
# !!     "name": "python"
# !!   },
# !!   "widgets": {
# !!     "application/vnd.jupyter.widget-state+json": {
# !!       "12dfc8f3be7043568f211ebd847d1ba4": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HBoxModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HBoxModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HBoxView",
# !!           "box_style": "",
# !!           "children": [
# !!             "IPY_MODEL_bbf8ce574f9a40d8b0e31cc374893927",
# !!             "IPY_MODEL_2029bd4e74f242328b24e604fb3832c1",
# !!             "IPY_MODEL_bd0d78d3d49b422fb5a2520d3e79469d"
# !!           ],
# !!           "layout": "IPY_MODEL_c9e18976325f48c295a1e1b5fa09fef7"
# !!         }
# !!       },
# !!       "bbf8ce574f9a40d8b0e31cc374893927": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_430774531dee4cf99010397bb5f445e6",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_89a4e4d1e0ad419aafe26916747ea69c",
# !!           "value": "100%"
# !!         }
# !!       },
# !!       "2029bd4e74f242328b24e604fb3832c1": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "FloatProgressModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "FloatProgressModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "ProgressView",
# !!           "bar_style": "success",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_7b5e46be957d48779549bc684cb73cae",
# !!           "max": 100,
# !!           "min": 0,
# !!           "orientation": "horizontal",
# !!           "style": "IPY_MODEL_9ab76575971f493a98af39cad78244c1",
# !!           "value": 100
# !!         }
# !!       },
# !!       "bd0d78d3d49b422fb5a2520d3e79469d": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_a96fea00e61a4a0492d5efe37da99804",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_ccaf4675d9f344be8dc4c80baccf9e2e",
# !!           "value": " 100/100 [00:19&lt;00:00,  5.07it/s]"
# !!         }
# !!       },
# !!       "c9e18976325f48c295a1e1b5fa09fef7": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "430774531dee4cf99010397bb5f445e6": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "89a4e4d1e0ad419aafe26916747ea69c": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "7b5e46be957d48779549bc684cb73cae": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "9ab76575971f493a98af39cad78244c1": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "ProgressStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "ProgressStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "bar_color": null,
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "a96fea00e61a4a0492d5efe37da99804": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "ccaf4675d9f344be8dc4c80baccf9e2e": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       }
# !!     }
# !!   }
# !! }}
