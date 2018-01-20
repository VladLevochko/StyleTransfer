import argparse
import os

from image_utils import load_image
from scipy.misc import imsave
from style_transfer_slim import StyleTransfer
from tensorflow.contrib.slim.nets import inception, resnet_v1, vgg


def run_with_inception_v1(content_image, style_image, config):
    config["style_layers"] = []
    config["style_weights"] = []
    config["content_layers"] = []

    sf = StyleTransfer(inception.inception_v1_base, config)
    generated_image = sf.transfer(content_image, style_image)

    save_generated_image(generated_image, config["summary_directory"])


def run_with_resnet(content_image, style_image, config):
    config["style_layers"] = []
    config["style_weights"] = []
    config["content_layers"] = []

    sf = StyleTransfer(resnet_v1.resnet_v1_101, config)
    generated_image = sf.transfer(content_image, style_image)

    save_generated_image(generated_image, config["summary_directory"])


def run_with_nasnet(content_image, style_image, config):
    config["style_layers"] = []
    config["style_weights"] = []
    config["content_layers"] = []

    sf = StyleTransfer(vgg.vgg_16, config)
    generated_image = sf.transfer(content_image, style_image)

    save_generated_image(generated_image, config["summary_directory"])


def run_with_vgg16(content_image, style_image, config):
    config["style_layers"] = ["conv1_1", "conv3_1"]
    config["style_weights"] = [2e-1, 2e-1]
    config["content_layers"] = ["conv4_1"]
    config["checkpoint"] = "checkpoints/vgg_16.ckpt"

    sf = StyleTransfer(vgg.vgg_16, config)
    generated_image = sf.transfer(content_image, style_image)

    save_generated_image(generated_image, config["summary_directory"])


def save_generated_image(image, summary_directory):
    files = os.listdir(summary_directory)
    target_directory = sorted(files)[-1]
    image_name = os.path.join(target_directory, "image.png")
    imsave(image_name, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["vgg16", "inception", "resnet", "nasnet"],
                        help="model which will be used for feature extraction")
    parser.add_argument("--content_image", required=True, help="path to the content image")
    parser.add_argument("--style_image", required=True, help="path to the style image")
    parser.add_argument("--iterations", default=500, type=int, help="number of iterations")
    parser.add_argument("--summary_dir", required=True, help="directory where summaries will be stored")
    parser.add_argument("--lr", default=1e-1, help="learning rate")
    parser.add_argument("--cl_weight", default=1, help="content loss weight")
    parser.add_argument("--sl_weight", default=1, help="style loss weight")
    args = parser.parse_args()

    config = {
        "num_iterations": int(args.iterations),
        "summary_directory": args.summary_dir,
        "learning_rate": float(args.lr),
        "content_loss_weight": float(args.cl_weight),
        "style_loss_weight": float(args.sl_weight)
    }

    content_image = load_image(args.content_image, 256)
    style_image = load_image(args.style_image, 256)

    if args.model == "inception":
        run_with_inception_v1(content_image, style_image, config)
    elif args.model == "resnet":
        run_with_resnet(content_image, style_image, config)
    elif args.model == "nasnet":
        run_with_nasnet(content_image, style_image, config)
    elif args.model == "vgg16":
        run_with_vgg16(content_image, style_image, config)
