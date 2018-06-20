"""Transform an image into the target style

Usage:
  transform.py MODEL INPUT OUTPUT [--resize=<size>]
  transform.py -h | --help

Load MODEL and use it to transform INPUT into OUTPUT.

Arguments:
  MODEL       .pth Pytorch state dict
  INPUT       input image file
  OUTPUT      output file path

Options:
  -h --help         Show this screen
  --resize=<size>   Resize shorter edge of the input [default: None]
"""
from docopt import docopt
import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

from fast_neural_style.utils import recover_image, tensor_normalizer
from fast_neural_style.transformer_net import TransformerNet


def load_model(model_file):
    transformer = TransformerNet()
    transformer.load_state_dict(torch.load(model_file))
    return transformer


def load_and_preprocess(image_file, size):
    img = Image.open(image_file).convert('RGB')
    if size != "None":
        transform = transforms.Compose([
            transforms.Resize(int(size)),
            transforms.ToTensor(),
            tensor_normalizer()])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            tensor_normalizer()])

    img_tensor = transform(img).unsqueeze(0)
    return img_tensor


def transform(model_file, image_file, target_path, size):
    transformer = load_model(model_file)
    img_var = load_and_preprocess(image_file, size)
    img_output = transformer(img_var)
    output_img = Image.fromarray(
        recover_image(img_output.data.numpy())[0])
    output_img.save(target_path)


if __name__ == "__main__":
    args = docopt(__doc__)
    print(args)
    transform(args["MODEL"], args["INPUT"], args["OUTPUT"], args["--resize"])
