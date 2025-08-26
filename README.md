# wafer-defects-ml
Physics-inspired wafer defect detection: code and recipes to fine-tune a multi-label ResNet18, and classify particles, scratches, and refractive index halos based on short wave infrared upconverted microscope images.
<p align="center">
  <img src="assets/download (1).png" width="85%">
</p>

<p align="center">
  <img src="assets/output 1.png" width="85%">
</p>

**Left column:** raw SWIR input and predicted output (labels shown under each).  
**Right column:** brightness-enhanced versions for human visibility.  
**Note:** SWIR captures are inherently low-brightness; some scratches are only visible via the model’s attention (e.g., Grad-CAM) and not to the naked eye.
