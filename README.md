# Deniose-Image-Using-GAN

## pre-processing
1. open terminal in the project location
    
    ```
    mkdir ./Face Dataset/input
    mkdir ./Face Dataset/target
    mkdir models
    ```

    Above commands create two folders inside ./Face Dataset/ folder, one is for input to GAN and one more is for target images
 
 2. run the preprocess.py file to add noise to images, Here we are adding noise to images and giving noise images images as 
 input images and original images as target images, So GAN will train on these kind of images and when we gave new noise image
 GAN will generate noiseless image.
 
    ```
    python preprocess.py --original './Face Dataset/original/' --input './Face Dataset/input' 
    --target 'Face Dataset/target/'
    ``` 
 ## Training
