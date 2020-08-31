from PIL import Image
import torchvision.transforms as transforms


class ImageUtilities(object):
    @staticmethod
    def image_resizer(height, width, interpolation=Image.BILINEAR):
        return transforms.Resize((height, width), interpolation=interpolation)