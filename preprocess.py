# python preprocess.py --original './Face Dataset/original/' --input './Face Dataset/input' --target 'Face Dataset/target/'

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--original", required=True,
	help="folder path to original images")
ap.add_argument("-i", "--input", required=True,
	help="folder path to input images")
ap.add_argument("-t", "--target", required=True,
	help="folder path to target image")
args = vars(ap.parse_args())


def add_noise(image):
    row,col,ch = image.shape
    s_vs_p = 0.5
    amount = 0.5
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
          for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
          for i in image.shape]
    out[coords] = 0
    return out

def main(original_images_path,input_images_path,target_images_path):
    print('----------------------PreProcessing----------------------')
    for i,img in enumerate(os.listdir(original_images_path)):
        image = cv2.imread(os.path.join(original_images_path,img))
        name = str(i)+'.jpg'
        image_noise = add_noise(image)
        cv2.imwrite(os.path.join(target_images_path, name), image)
        cv2.imwrite(os.path.join(input_images_path, name),image_noise)
    
    print('----------------------Completed----------------------')
    

if __name__ == "__main__":
    original_images_path = args["original"]
    input_images_path = args["input"]
    target_images_path = args["target"]
    main(original_images_path, input_images_path, target_images_path)