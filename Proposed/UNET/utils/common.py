# import cv2
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

	
def show_image(origin, reconst, display=2, epoch=0, path=None):
    plt.figure()
<<<<<<< HEAD
    for i in range(display):
        plt.subplot(2,2,(i*2)+1)                                                                                                      
        plt.imshow(to_pil_image(origin[i]))
        plt.title('Origin')
        
        plt.subplot(2,2,(i*2)+2)
        plt.imshow(to_pil_image(reconst[i]))
        plt.title('Reconstruction')
=======
    plt.subplot(1,2,1)                                                                                                      
    plt.imshow(to_pil_image(origin[0]))
    plt.title('Origin')
	
    plt.subplot(1,2,2)
    plt.imshow(to_pil_image(reconst[0]))
    plt.title('Reconstruction')
>>>>>>> a0d0566c23dc59e4d6d839ebd8fd86995ebf71ed

    plt.tight_layout()
    plt.savefig(path)
