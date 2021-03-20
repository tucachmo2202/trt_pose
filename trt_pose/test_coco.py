import cv2 
import PIL.Image
import numpy as np
import torch


image = PIL.Image.open("/home/manhas/Pictures/1.jpg")
image = image.resize((1536//2, 2048//2))
image_shape = [2048//2, 1536//2]
mask = np.ones(image_shape, dtype=np.uint8)*255
print(mask.shape)
mask[746//2:2044//2, 427//2:1008//2] = 0
cv2.imshow("as", mask)
cv2.waitKey()
cv2.destroyAllWindows()
mask = PIL.Image.fromarray(mask)

def get_quad(angle, translation, scale, aspect_ratio=1.0):
    if aspect_ratio > 1.0:
        # width > height =>
        # increase height region
        quad = np.array([
            [0.0, 0.5 - 0.5 * aspect_ratio],
            [0.0, 0.5 + 0.5 * aspect_ratio],
            [1.0, 0.5 + 0.5 * aspect_ratio],
            [1.0, 0.5 - 0.5 * aspect_ratio],
            
        ])
    elif aspect_ratio < 1.0:
        # width < height
        quad = np.array([
            [0.5 - 0.5 / aspect_ratio, 0.0],
            [0.5 - 0.5 / aspect_ratio, 1.0],
            [0.5 + 0.5 / aspect_ratio, 1.0],
            [0.5 + 0.5 / aspect_ratio, 0.0],
            
        ])
    else:
        quad = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ])
        
    quad -= 0.5

    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    quad = np.dot(quad, R)

    quad -= np.array(translation)
    quad /= scale
    quad += 0.5
    
    return quad


def transform_image(image, size, quad):
    new_quad = np.zeros_like(quad)
    new_quad[:, 0] = quad[:, 0] * image.size[0]
    new_quad[:, 1] = quad[:, 1] * image.size[1]
    
    new_quad = (new_quad[0][0], new_quad[0][1],
            new_quad[1][0], new_quad[1][1],
            new_quad[2][0], new_quad[2][1],
            new_quad[3][0], new_quad[3][1])
    
    return image.transform(size, PIL.Image.QUAD, new_quad)


def transform_points_xy(points, quad):
    p00 = quad[0]
    p01 = quad[1] - p00
    p10 = quad[3] - p00
    p01 /= np.sum(p01**2)
    p10 /= np.sum(p10**2)
    
    A = np.array([
        p10,
        p01,
    ]).transpose()
    
    return np.dot(points - p00, A)


# def transform_peaks(counts, peaks, quad):
#     newpeaks = peaks.clone().numpy()
#     C = counts.shape[0]
#     for c in range(C):
#         count = int(counts[c])
#         newpeaks[c][0:count] = transform_points_xy(newpeaks[c][0:count][:, ::-1], quad)[:, ::-1]
#     return torch.from_numpy(newpeaks)

random_angle = [-0.15, 0.15]
random_scale = [0.5, 1.5]
random_translate = [-0.15, 0.15] 

shiftx = float(torch.rand(1)) * (random_translate[1] - random_translate[0]) + random_translate[0]
shifty = float(torch.rand(1)) * (random_translate[1] - random_translate[0]) + random_translate[0]
scale = float(torch.rand(1)) * (random_scale[1] - random_scale[0]) + random_scale[0]
angle = float(torch.rand(1)) * (random_angle[1] - random_angle[0]) + random_angle[0]

keep_aspect_ratio = False
if keep_aspect_ratio:
    ar = float(image.width) / float(image.height)
    quad = get_quad(angle, (shiftx, shifty), scale, aspect_ratio=ar)  
else:
    quad = get_quad(angle, (shiftx, shifty), scale, aspect_ratio=1.0)

target_shape = [256, 256]
# image.show()
# mask.show()
image = transform_image(image, (image_shape[1], image_shape[0]), quad)
mask = transform_image(mask, (target_shape[1], target_shape[0]), quad)
image.show()
# mask.show()

import json 
import pprint
from trt_pose.coco import CocoDataset, CocoHumanPoseEval
import tqdm


config_file = "/home/manhas/Desktop/PoseEstimation/trt_pose/tasks/human_pose/experiments/resnet50_baseline_att_320_256_A.json"
with open(config_file, 'r') as f:
    config = json.load(f)
    pprint.pprint(config)

test_dataset_kwargs = config["test_dataset"]
test_dataset = CocoDataset(**test_dataset_kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, **config["test_loader"])
for images, cmaps, masks in tqdm.tqdm(iter(test_loader)):
    print(len(images))
    print(len(cmaps))
    print(len(masks))
    image[0].show()
    break