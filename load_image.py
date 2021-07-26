import os
import sys
import random

def load_image_from_HMDB_video(root_path, action_class, img_index=None):
    
    second_path = os.path.join(root_path, action_class)

    directions = []
    for roots, dirs, files in os.walk(second_path):
        directions.append(roots)

    direction = random.randint(0, len(directions))
    package = directions[direction]
    
    list_file = os.listdir(package)
    _file = random.randint(0, len(list_file))
    file_name = None
    
    if img_index is None:
        file_name = list_file[_file]
    else:
        file_name = img_index

    file_name = os.path.join(package, file_name)

    return file_name


if __name__ == '__main__':

    root_path = "/data/guojie/HMDB51/VideoSeq"
    img_file = load_image_from_HMDB_video(root_path, 'talk', None)
    print(img_file)