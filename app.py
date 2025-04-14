"""
Neural Style Transfer Script
-----------------------------
This script accepts two images as input:
1. Content image – provides the structure/subject of the image.
2. Style image – provides the artistic style to be transferred.

The final result is saved as "styled_output.jpg".
Before running, ensure the following packages are installed:
    pip install torch==1.12.0 torchvision==0.13.0 pillow
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import copy

# Set device: use GPU if available; otherwise, fall back to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image size: larger size if GPU is available.
imsize = 512 if torch.cuda.is_available() else 128

# Transformation to load images: resize and convert to tensor.
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])

# Transformation to convert tensor back to PIL image for saving.
unloader = transforms.ToPILImage()

def image_loader(image_path):
    """Load an image and convert it to a torch tensor."""
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)  # add batch dimension: (1, C, H, W)
    return image.to(device, torch.float)

def gram_matrix(input_tensor):
    """
    Compute the Gram Matrix for the given tensor.
    The Gram matrix represents the correlations between the feature maps.
    """
    a, b, c, d = input_tensor.size()  # a: batch size, b: number of feature maps, c,d: dimensions
    features = input_tensor.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

# Module to compute content loss.
class ContentLoss(nn.Module):
    def __init__(self, target, weight=1):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.weight = weight
        self.loss = 0

    def forward(self, input):
        self.loss = self.weight * nn.functional.mse_loss(input, self.target)
        return input

# Module to compute style loss using Gram matrices.
class StyleLoss(nn.Module):
    def __init__(self, target_feature, weight=1e6):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.weight = weight
        self.loss = 0

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = self.weight * nn.functional.mse_loss(G, self.target)
        return input

# Normalization module needed by the pre-trained CNN.
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # Reshape to (C, 1, 1) for broadcasting.
        self.mean = torch.tensor(mean).to(device).view(-1, 1, 1)
        self.std  = torch.tensor(std).to(device).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=['conv_4'],
                               style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    """
    Create a new nn.Sequential model from the pre-trained CNN, inserting
    our custom content and style loss modules at the appropriate layers.
    """
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses   = []
    model = nn.Sequential(normalization)

    i = 0  # Incremental counter for the conv layers.
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)  # Use out-of-place ReLU.
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Trim the model after the last loss layer.
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]
    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    """Returns an optimizer that will update the input image."""
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1e6, content_weight=1):
    """
    Run the neural style transfer method, optimizing the input image to simultaneously
    match the content of the content image and the style of the style image.
    """
    print("Building the model and computing losses...")
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img)

    optimizer = get_input_optimizer(input_img)
    print("Starting optimization...")
    run = [0]
    while run[0] <= num_steps:
        
        def closure():
            input_img.data.clamp_(0, 1)  # Ensure pixel values are in [0, 1].
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
            
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            
            loss = style_score + content_score
            loss.backward()
            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}: Style Loss: {style_score.item():.4f} Content Loss: {content_score.item():.4f}")
            return loss
        
        optimizer.step(closure)
    
    input_img.data.clamp_(0, 1)
    return input_img

if __name__ == "__main__":
    # Ask the user for paths to the content and style images.
    content_path = input("Enter the path for the content image (e.g., C:\\path\\to\\content.jpg): ").strip()
    style_path   = input("Enter the path for the style image (e.g., C:\\path\\to\\style.jpg): ").strip()

    # Load the images.
    content_img = image_loader(content_path)
    style_img   = image_loader(style_path)
    
    if content_img.size() != style_img.size():
        print("Warning: Content and style images have different dimensions. For best results, use images of the same size.")

    # Use a copy of the content image as the initial input.
    input_img = content_img.clone()
    
    # Load a pre-trained VGG-19 model.
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    
    # VGG model normalization parameters.
    cnn_normalization_mean = [0.485, 0.456, 0.406]
    cnn_normalization_std  = [0.229, 0.224, 0.225]
    
    # Run the style transfer.
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, num_steps=300)
    
    # Convert the output tensor to a PIL image and save it.
    final_img = output.cpu().clone().squeeze(0)
    final_img = unloader(final_img)
    final_img.save("styled_output.jpg")
    print("The styled image is saved as 'styled_output.jpg'")
