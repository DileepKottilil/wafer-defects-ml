# wafer-defects-ml
Physics-inspired wafer defect detection: code and recipes to fine-tune a multi-label ResNet18, and classify particles, scratches, and refractive index halos based on short wave infrared upconverted microscope images.

THe below figures has 4 sub plots. The first column contains input (directly from experiments) and output images (from the trained resnet18 model). THe second column contains both the same images but brighter versions to show where the defects/contaminations are. With naked eye or camera, it is not possible to see the defects/contaminations from the experiments. 

The output classes of the model are written wlong with these subplots.

Fig 1
<p align="center">
  <img src="assets/download (1).png" width="85%">
</p>


Fig 2

<p align="center">
  <img src="assets/output1.png" width="85%">
</p>

**Left column:** raw SWIR input and predicted output (labels shown under each).  
**Right column:** brightness-enhanced versions for human visibility.  
**Note:** SWIR captures are inherently low-brightness; some scratches are only visible via the model’s attention and not to the naked eye.

The trained ResNet18 weights can be downloaded from the [Releases page](https://github.com/DileepKottilil/wafer-defects-ml/releases/tag/model).
Use your own images or images (wafer_defect_xxx.png) in the assets folder to test the model.

Use the below code to predict the defects of your input image. You need to write your own code to exactly reproduce the output as shown above. 


```# Single-image inference for wafer defect multi-label ResNet18
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Configuratoin
MODEL_PATH = "/content/resnet18_wafer_multilabel.pt"  # Change it to your model path
CLASSES = ["particle", "scratch", "ri"]
IMG_SIZE = 224
THRESH = 0.5   # you can adjust per-class later


eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# Model loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)  # architecture only
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt)
model = model.to(device).eval()

def predict_image(path, thresh=THRESH):
    img = Image.open(path).convert("RGB")
    x = eval_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
    pred_flags = [c for c,p in zip(CLASSES, probs) if p >= thresh]
    return img, dict(zip(CLASSES, map(float, probs))), pred_flags

# Running and visualization
def show_prediction(path, thresh=THRESH):
    img, prob_dict, pred_flags = predict_image(path, thresh)
    
    print("Image:", path)
    for c in CLASSES:
        print(f"{c:8s}: {prob_dict[c]:.3f}")
    print("Predicted (>= {:.2f}):".format(thresh), pred_flags if pred_flags else ["none"])

   
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis("off")
    
    title = ", ".join(pred_flags) if pred_flags else "no defect >= {:.2f}".format(thresh)
    plt.title(title)
    plt.show()

#Usage
# change this to your single image path. 
test_image_path = "/content/wafer_defect_dataset131/content/wafer_defect_dataset13/_manual_holdout/wafer_defect_196.png"
show_prediction(test_image_path)



