from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed

from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from matplotlib import pyplot

from utils.generator import Generator
from utils.discriminator import Discriminator
from utils.gan import GAN
import cv2
from numpy import savez_compressed


WIDTH = 256
HEIGHT = 256
DEPTH = 3
EPOCHS = 100

# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

# load all images in a directory into memory
def load_images(path,tar_path, size=(HEIGHT,WIDTH)):
    src_list, tar_list = list(), list()
    # enumerate filenames in directory, assume all are images
    for filename in listdir(path):
        # load and resize the image
        pixels = load_img(path + filename, target_size=size)
        tar_pixels = load_img(tar_path + filename, target_size=size)
        # convert to numpy array
        pixels = img_to_array(pixels)
        tar_pixels = img_to_array(tar_pixels)
        
        # split into satellite and map
        # sat_img, map_img = pixels[:, :256], pixels[:, 256:]
        src_list.append(pixels)
        tar_list.append(tar_pixels)
    return [asarray(src_list), asarray(tar_list)]

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB = dataset
	# choose random instances
	ix = randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

def summarize_performance(step, g_model, dataset, n_samples=3):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# plot real source images
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])
	# save plot to file
	filename1 = './results/plot_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = './models/model_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))
    

# train pix2pix model
def train(d_model, g_model, gan_model, dataset, n_epochs=EPOCHS, n_batch=1):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
        # summarize model performance
        if (i+1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model, dataset)

def main():
    # dataset path
    input_images_path = './Face Dataset/input/'
    target_images_path = './Face Dataset/target/'
    [src_images, tar_images] = load_images(input_images_path,target_images_path)
    print('Loaded: ', src_images.shape, tar_images.shape)
    # save as compressed numpy array
    filename = './loaded_data/data.npz'
    savez_compressed(filename, src_images, tar_images)
    print('Saved dataset: ', filename)

    dataset = load_real_samples('./loaded_data/data.npz')
    print('Loaded', dataset[0].shape, dataset[1].shape)

    # image_shape = (HEIGHT,WIDTH,DEPTH)
    # define input shape based on the loaded dataset
    # image_shape = src_images[0].shape[1:]
    # define the models
    d_model = Discriminator().define_discriminator()
    g_model = Generator().define_generator()
    # define the composite model
    gan_model = GAN().define_gan(g_model, d_model)
    # train model
    train(d_model, g_model, gan_model, dataset)
    

if __name__ == "__main__":
    main()