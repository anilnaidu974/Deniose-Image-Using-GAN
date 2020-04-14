# python test.py --image 'path/to/input.png' --model './models/model.h5'

from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
import argparse
import matplotlib.image as mpimg


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input images")
ap.add_argument("-m", "--model", required=True,
	help="path to saved model")
args = vars(ap.parse_args())

# load an image
def load_op_image(filename, size=(256,256)):
    # load image with the preferred size
    pixels = load_img(filename, target_size=size)
    # convert to numpy array
    pixels = img_to_array(pixels)
    # scale from [0,255] to [-1,1]
    pixels = (pixels - 127.5) / 127.5
    # reshape to 1 sample
    pixels = expand_dims(pixels, 0)
    return pixels

def main(image_path,model_path):
    # load source image
    src_image = load_op_image(image_path)
    print('Loaded', src_image.shape)
    # load model
    model = load_model(model_path)
    # generate image from source
    gen_image = model.predict(src_image)
    # scale from [-1,1] to [0,1]
    temp = gen_image[0]
    img = (temp + 1) / 2.0

    mpimg.imsave("output.png", img)
    print('******************************************************************************')

if __name__ == "__main__":
    image_pth = args['image']
    model_path = args['model']
    main(image_pth, model_path)