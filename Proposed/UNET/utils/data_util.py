import os 
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_imagelist(domain):
    path = f'C:\\Users\\MINJU\\UDA_res\\Dataset\\domain_adaptation_images\\{domain}\\images' 
    total = 0
    trains = []
    tests  = []
    assert os.path.isdir(path), '%s is not a valid directory' % path
    for root, _, fnames in sorted(os.walk(path)):
        total += len(fnames)
        train_idx = int(len(fnames)*0.8)
        random.shuffle(fnames)
        for idx,fname in enumerate(fnames):
            if idx < train_idx:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    trains.append(path)
            else:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    tests.append(path)
    return trains, tests

