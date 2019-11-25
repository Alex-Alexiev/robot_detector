import os, sys

import download_images
import transform_image_resolution
import xml_to_csv

def main():
    download_images.download_images()
    # os.rename('../images/frc', '../images/train')
    # os.rename('../images/frc robot', '../images/test')

    transform_image_resolution.rescale_images('../images/frc/', (800, 600))
    transform_image_resolution.rescale_images('../images/frc robot/', (800, 600)) 
    

if __name__ == "__main__":
    main()






