# Image Processing Scripts #

## To Retrieve Images and Clean up for RGB Format ##
Execute following scripts in order:
1. run get_fake_images.py
2. get_real_images.py
    NOTE: you will need to manually download the real images before running this script. Further instructions can be found in the script.
3. cleanup_images.py

* All these images get saved to directory: /image_data

## To Clean up Images for FFT and Mag Spec Format ##
1. run batch_fft_preprocess.py

* All these images get saved to directory: /frequency_data

To run analysis on FFT and mag spec images: frequency_spectrum.py
