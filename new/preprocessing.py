from PIL import Image, ImageFilter
import os

def PIL_erosion(image, size):
    image = image.filter(ImageFilter.MinFilter(size))
    return image

def PIL_dilation(image, size):
    image = image.filter(ImageFilter.MaxFilter(size))
    return image

def PIL_opening(image, size):
    image = PIL_erosion(image, size)
    image = PIL_dilation(image, size)
    return image

def PIL_thresholding(image, threshold):
    image = image.point(lambda p: 255 if p > threshold else 0)
    return image

def main():
    size = 3
    threshold = 200
    image_dir = "data/train"
    save_dir = "thresholding/train"

    if (not os.path.exists(save_dir)):
        os.makedirs(save_dir)

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path)

        # image = PIL_opening(image, size)
        image = PIL_thresholding(image, threshold)

        image_save_path = os.path.join(save_dir, image_name)
        image.save(image_save_path)

if __name__ == "__main__":
    main()