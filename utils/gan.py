from keras.models import Input
from keras.models import Model
from keras.optimizers import Adam



class GAN():

    def __init__(self,width = 256, height = 256, depth = 3):
        self.width = width
        self.height = height
        self.depth = depth



    def define_gan(self,g_model, d_model):
        image_shape=(self.height,self.width,self.depth)
        # make weights in the discriminator not trainable
        d_model.trainable = False
        # define the source image
        in_src = Input(shape=image_shape)
        # connect the source image to the generator input
        gen_out = g_model(in_src)
        # connect the source input and generator output to the discriminator input
        dis_out = d_model([in_src, gen_out])
        # src image as input, generated image and classification output
        model = Model(in_src, [dis_out, gen_out])
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
        return model