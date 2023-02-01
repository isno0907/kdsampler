import torchvision
import os
import pickle
import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import mmcv
# wrong_video_list = []

manager = multiprocessing.Manager()
wrong_video_list = manager.list()

# counter = manager.Value('i', 0)
def check(video):
    img_list = os.listdir(os.path.join(root,video))
    print(video, "processing", end=" ")
    flag = 1 
    # counter += 1
    for i in range(len(img_list)):
        try:
            torchvision.io.read_image(os.path.join(root,video, img_list[i]))
            # torchvision.io.read_image(os.path.join('/data/dataset/mmaction2/data/kinetics400/new_rawframes_train/O1XGEYhGEfA_000028_000038/img_00013.jpg'))
        except:
            wrong_video_list.append(video)
            flag = 0
            break
    if flag:
        print("Done")
    else:
        print("Failed")
    # prog_bar.update()

     
root  = 'data/kinetics400/new_rawframes_train'
#video_list = os.listdir(root)
video_list = np.load("wronglist.npy")
pool = Pool(processes=32)
pool.map(check, video_list)
pool.close()
pool.join()
# for video in tqdm(video_list):
#     print(video, "processing", end=" ")
#     flag = 1 
#     img_list = os.listdir(os.path.join(root,video))
#     for i in range(len(img_list)):
#         try:
#             torchvision.io.read_image(os.path.join(root,video, img_list[i]))
#             # torchvision.io.read_image(os.path.join('/data/dataset/mmaction2/data/kinetics400/new_rawframes_train/O1XGEYhGEfA_000028_000038/img_00013.jpg'))
#         except:
#             wrong_video_list.append(video)
#             flag = 0
#             break
#     if flag:
#         print("Done")
#     else:
#         print("Failed")
# barrier = multiprocessing.Barrier(10)
# barrier.wait()
print(wrong_video_list)
wl = np.asarray(wrong_video_list)
np.save("wrongwrongwronglist", wl)
