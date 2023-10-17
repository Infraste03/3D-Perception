
# Import the DPTForSemanticSegmentation from the Transformers library
from transformers import DPTForSemanticSegmentation

# Create the DPTForSemanticSegmentation model and load the pre-trained weights
# The "Intel/dpt-large-ade" model is a large-scale model trained on the ADE20K dataset
model = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade")



from PIL import Image
import requests
import torch.utils.data as data



# URL of the image to be downloaded

url = 'https://storage.googleapis.com/kagglesdsdata/datasets/1073535/1806921/001/camera/back_camera/00.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20231016%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20231016T121757Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=7aa1a47fca98208fc13ffb0debe126aa5bb1422e09cd3f4b06a4d2e25bf14df9a143f201e3f36bcbd3ce9e6ae2e6b9b05cf2fa4612cc224797b0ff86b66682122874c40431a2f86faa3bd06165e7486d26e6d7c756115bae03e21c0f82b143b28eb9d2dfd80343eb4ba47b23517e5da42f97f2dcea44d613975a083f8722ca939b1ddef29400c69cf138e215229779cc95a9248c83e86f834a13df80cc96c8f0213cd780da99494ac98b1aaa1ce6a70a03ee1914983bea4e3268c49f02e53598866906bc4e882c169099de33b1aafdb53eeb612c5d210ee9b8d0f7a7b07b686c1e8afdda77eefddc9dced776367dfb568d43a9f68c4d7cae1e5169618ab82cd6'

import torchvision.transforms as transforms
# The `stream=True` parameter ensures that the response is not immediately downloaded, but is kept in memory
response = requests.get(url, stream=True)

# Create the Image class
image = Image.open(response.raw)
image_test = Image.open('C:/Users/fstef/Desktop/3D/Progetto/005/camera/front_camera/06.jpg')
mask = Image.open('C:/Users/fstef/Desktop/3D/Progetto/006Mask.jpg')
transform = transforms.ToTensor()
image_tensor = transform(image)
mask_tensor = transform(mask)
from  torch.utils.data import  Dataset
import torch.utils.data as data





# Display image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop

# Set the desired height and width for the input image
net_h = net_w = 480

# Define a series of image transformations
transform = Compose([
        # Resize the image
        Resize((net_h, net_w)),
        #RandomCrop(227,padding=4),
        # Convert the image to a PyTorch tensor
        ToTensor(),
        # Normalize the image
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
# Transform input image
pixel_values = transform(image)

pixel_values = pixel_values.unsqueeze(0)

import torch

# Disable gradient computation
with torch.no_grad():
    # Perform a forward pass through the model
    outputs = model(pixel_values)
    # Obtain the logits (raw predictions) from the output
    logits = outputs.logits
import torch

# Interpolate the logits to the original image size
prediction = torch.nn.functional.interpolate(
    logits,
    size=image.size[::-1],  # Reverse the size of the original image (width, height)
    mode="bilinear",
    align_corners=False
)

# Convert logits to class predictions
prediction = torch.argmax(prediction, dim=1) + 1

# Squeeze the prediction tensor to remove dimensions
prediction = prediction.squeeze()

# Move the prediction tensor to the CPU and convert it to a numpy array
prediction = prediction.cpu().numpy()

from PIL import Image

# Convert the prediction array to an image
predicted_seg = Image.fromarray(prediction.squeeze().astype('uint8'))
adepallete = [
    255, 255, 250, 255, 255, 255, 255, 250, 255, 255, 250,255, 255, 250, 255, 255, 250, 255, 255, 250, 255, 255, 250, 140,
    204, 5, 255, 230, 230, 230, 4, 250, 7, 224, 5, 255, 235, 255, 7, 150, 5, 61, 120, 120, 70, 8, 255, 51,
    255, 6, 82, 143, 255, 140, 204, 255, 4, 255, 51, 7, 204, 255, 255, 204, 255, 255, 153, 6, 51, 255, 235,
    12, 160, 150, 20, 0, 255, 63, 255, 0, 255, 255, 0, 255, 255, 204, 255, 255, 0, 153, 255, 0, 0, 255, 255,
    0, 255, 184, 184, 0, 31, 255, 0, 255, 230, 224, 255, 255, 128, 255, 212, 128, 74, 255, 128, 255, 1, 255,
    255, 0, 228, 255, 0, 0, 160, 255, 0, 255, 0, 255, 255, 0, 0, 0, 255, 255, 255, 0, 255, 255, 255, 0, 255,
    255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255
]
# Apply the color map to the predicted segmentation image
predicted_seg.putpalette(adepallete)

# Blend the original image and the predicted segmentation image
out = Image.blend(image, predicted_seg.convert("RGB"), alpha=0.7)

out.show()

