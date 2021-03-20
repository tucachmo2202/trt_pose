import torch
import torch.utils.data
import torch.nn
import os
import PIL.Image
import json
import tqdm
import trt_pose
import trt_pose.plugins
import glob
import torchvision.transforms.functional as FT
import numpy as np
from trt_pose.parse_objects import ParseObjects
import pycocotools
import pycocotools.coco
import pycocotools.cocoeval
import torchvision


''' skeleton: [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [18, 1], [18, 6], [18, 7], [18, 12], [18, 13]]
    topology: tensor([[ 0,  1, 15, 13],
        [ 2,  3, 13, 11],
        [ 4,  5, 16, 14],
        [ 6,  7, 14, 12],
        [ 8,  9, 11, 12],
        [10, 11,  5,  7],
        [12, 13,  6,  8],
        [14, 15,  7,  9],
        [16, 17,  8, 10],
        [18, 19,  1,  2],
        [20, 21,  0,  1],
        [22, 23,  0,  2],
        [24, 25,  1,  3],
        [26, 27,  2,  4],
        [28, 29,  3,  5],
        [30, 31,  4,  6],
        [32, 33, 17,  0],
        [34, 35, 17,  5],
        [36, 37, 17,  6],
        [38, 39, 17, 11],
        [40, 41, 17, 12]], dtype=torch.int32)

Trong topology, mỗi đầu vào là [k_i, k_j, c_a, c_b] là kênh của paf tương ứng với chiều i,j
và c_a là c_b là loại của chi
connections - int - NxKx2xM, đồ thị kết nối.
Kết nối laoij 1 và nguồn chi chỉ số 5, chỉ số chi đích là 3 = connections[0][1][0][5]
hoặc kết nối laoij 1 và chỉ số chi đích là 3, chỉ số chi nguồn là 5 là connections[0][1][1][3],
đồ thị đại diện là thừa, cả 2 hướng được thêm vào cho thời gian tra cứu k đổi. Nếu 
k có kết nối tồn tại cho nguồn hoặc đích, nó trả về -1.
'''
        
def coco_category_to_topology(coco_category):
    """Gets topology tensor from a COCO category
    """
    skeleton = coco_category['skeleton']
    K = len(skeleton)
    topology = torch.zeros((K, 4)).int()
    for k in range(K):
        topology[k][0] = 2 * k
        topology[k][1] = 2 * k + 1
        topology[k][2] = skeleton[k][0] - 1
        topology[k][3] = skeleton[k][1] - 1
    return topology

#Trả về tên của các điểm keypoints
def coco_category_to_parts(coco_category):
    """Gets list of parts name from a COCO category
    """
    return coco_category['keypoints']


def coco_annotations_to_tensors(coco_annotations,
                                image_shape,
                                parts,
                                topology,
                                max_count=100):
    """Gets tensors corresponding to peak counts, peak coordinates, and peak to peak connections
    """
    annotations = coco_annotations
    C = len(parts)   # C = 17
    K = topology.shape[0]
    M = max_count
    IH = image_shape[0]  #image_shape là shape của ảnh gốc
    IW = image_shape[1]
    counts = torch.zeros((C)).int()
    peaks = torch.zeros((C, M, 2)).float()
    visibles = torch.zeros((len(annotations), C)).int()
    # connections = -torch.ones((K, 2, M)).int()

    for ann_idx, ann in enumerate(annotations):

        kps = ann['keypoints']

        # add visible peaks
        for c in range(C):

            x = kps[c * 3]
            y = kps[c * 3 + 1]
            visible = kps[c * 3 + 2]

            if visible:
                peaks[c][counts[c]][0] = (float(y) + 0.5) / (IH + 1.0)
                peaks[c][counts[c]][1] = (float(x) + 0.5) / (IW + 1.0)
                counts[c] = counts[c] + 1
                visibles[ann_idx][c] = 1

        # for k in range(K):
        #     c_a = topology[k][2]
        #     c_b = topology[k][3]
        #     if visibles[ann_idx][c_a] and visibles[ann_idx][c_b]:
        #         connections[k][0][counts[c_a] - 1] = counts[c_b] - 1
        #         connections[k][1][counts[c_b] - 1] = counts[c_a] - 1

    return counts, peaks


# def coco_annotations_to_mask_bbox(coco_annotations, image_shape):
#     mask = np.ones(image_shape, dtype=np.uint8) # mask khởi tạo bằng mảng các giá trị 1
#     #bbox của coco có định dạng [top left x position, top left y position, width, height]
#     for ann in coco_annotations:
#         if 'num_keypoints' not in ann or ann['num_keypoints'] == 0: #nếu không có keypoints nào thì mask của bbox đó thay bằng giá trị 0
#             bbox = ann['bbox']
#             x0 = round(bbox[0])
#             y0 = round(bbox[1])
#             x1 = round(x0 + bbox[2])
#             y1 = round(y0 + bbox[3])
#             mask[y0:y1, x0:x1] = 0
#     return mask
            

def convert_dir_to_bmp(output_dir, input_dir):
    files = glob.glob(os.path.join(input_dir, '*.jpg'))
    for f in files:
        new_path = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(f))[0] + '.bmp')
        img = PIL.Image.open(f)
        img.save(new_path)

        
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


def transform_peaks(counts, peaks, quad):
    newpeaks = peaks.clone().numpy()
    C = counts.shape[0]
    for c in range(C):
        count = int(counts[c])
        newpeaks[c][0:count] = transform_points_xy(newpeaks[c][0:count][:, ::-1], quad)[:, ::-1]
    return torch.from_numpy(newpeaks)


class CocoDataset(torch.utils.data.Dataset):
    def __init__(self,
                 images_dir,
                 annotations_file,
                 category_name,
                 image_shape,
                 target_shape,
                 is_bmp=False,
                 stdev=0.02,
                 use_crowd=True,
                 min_area=0.0,
                 max_area=1.0,
                 max_part_count=100,
                 random_angle=(0.0, 0.0),
                 random_scale=(1.0, 1.0),def coco_annotations_to_mask_bbox(coco_annotations, image_shape):
#     mask = np.ones(image_shape, dtype=np.uint8) # mask khởi tạo bằng mảng các giá trị 1
#     #bbox của coco có định dạng [top left x position, top left y position, width, height]
#     for ann in coco_annotations:
#         if 'num_keypoints' not in ann or ann['num_keypoints'] == 0: #nếu không có keypoints nào thì mask của bbox đó thay bằng giá trị 0
#             bbox = ann['bbox']
#             x0 = round(bbox[0])
#             y0 = round(bbox[1])
#             x1 = round(x0 + bbox[2])
#             y1 = round(y0 + bbox[3])
#             mask[y0:y1, x0:x1] = 0
#     return mask
                 random_translate=(0.0, 0.0),
                 transforms=None,
                 keep_aspect_ratio=False): #tham số keep_aspect_ratio để chỉ rằng ảnh sẽ resize padding??

        self.keep_aspect_ratio = keep_aspect_ratio
        self.transforms=transforms
        self.is_bmp = is_bmp
        self.images_dir = images_dir
        self.image_shape = image_shape
        self.target_shape = target_shape
        self.stdev = stdev
        self.random_angle = random_angle
        self.random_scale = random_scale
        self.random_translate = random_translate
        
        tensor_cache_file = annotations_file + '.cache'
        
        if tensor_cache_file is not None and os.path.exists(tensor_cache_file):
            print('Cachefile found.  Loading from cache file...')
            cache = torch.load(tensor_cache_file)
            self.counts = cache['counts']
            self.peaks = cache['peaks']
            # self.connections = cache['connections']
            self.topology = cache['topology']
            self.parts = cache['parts']
            self.filenames = cache['filenames']
            self.samples = cache['samples']
            return
            
        with open(annotations_file, 'r') as f:
            data = json.load(f)

        cat = [c for c in data['categories'] if c['name'] == category_name][0]
        cat_id = cat['id']

''' 
Thẻ images chứa các thông tin như trong ví dụ dưới đây: 
    "images": [
        {
            "id": 0,
            "license": 1,
            "file_name": "0001.jpg",
            "height": 275,
            "width": 490,
            "date_captured": "2020-07-20T19:39:26+00:00"
        }
    ],
'''         
        samples = {}
        for ann in data['annotations']:

            # filter by category
            if ann['category_id'] != cat_id:
                continue

            # filter by crowd
            # if not use_crowd and ann['iscrowd']:
            #     continue

            #loai bo anh k co keypoints
            if 'num_keypoints' not in ann or ann['num_keypoints'] == 0:
                continue

            img_id = ann['image_id']
            height = img['height']
            width = img['width']
            # area = ann['area']

            # filter by object area
            # normalized_area = float(area) / float(height * width)
            # if normalized_area < min_area or normalized_area > max_area:
            #     continue

            # add metadata
            if img_id not in samples:
                sample = {}
                sample['img'] = img
                sample['anns'] = [ann]
                samples[img_id] = sample
            else:
                samples[img_id]['anns'] += [ann]
                
        # generate tensors
        self.topology = coco_category_to_topology(cat)
        self.parts = coco_category_to_parts(cat)

        N = len(samples)
        C = len(self.parts)
        K = self.topology.shape[0]
        M = max_part_count

        print('Generating intermediate tensors...')
        self.counts = torch.zeros((N, C), dtype=torch.int32)
        self.peaks = torch.zeros((N, C, M, 2), dtype=torch.float32)
        # self.connections = torch.zeros((N, K, 2, M), dtype=torch.int32)
        self.filenames = []
        self.samples = []
        
        for i, sample in tqdm.tqdm(enumerate(samples.values())):
            filename = sample['img']['file_name']
            self.filenames.append(filename)
            # image_shape = (sample['img']['height'], sample['img']['width'])
            # bbox của coco có định dạng [top left x position, top left y position, width, height]
            count_array = []
            peak_array = []
            for j, ann in enumerate(sample['anns']):
                image_shape = ann['bbox'][3], ann['bbox'][2]
                new_ann = ann.deepcopy()
                for i in range len(ann['keypoints']//3):
                    new_ann['keypoints'][3*i] -= new_ann['bbox'][0]
                    new_ann['keypoints'][3*i+1] -= new_ann['bbox'][1]
                counts_i, peaks_i= coco_annotations_to_tensors([new_ann], image_shape, self.parts, self.topology)
                count_array.append(counts_i)
                peak_array.append(peaks_i)
            self.counts[i] = count_array
            self.peaks[i] = peak_array
            self.samples += [sample]

        if tensor_cache_file is not None:
            print('Saving to intermediate tensors to cache file...')
            torch.save({
                'counts': self.counts,
                'peaks': self.peaks,
                # 'connections': self.connections,
                'topology': self.topology,
                'parts': self.parts,
                'filenames': self.filenames,
                'samples': self.samples
            }, tensor_cache_file)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        if self.is_bmp:
            filename = os.path.splitext(self.filenames[idx])[0] + '.bmp'
        else:
            filename = os.path.splitext(self.filenames[idx])[0] + '.jpg'

        image = PIL.Image.open(os.path.join(self.images_dir, filename))
        

        images = []
        cmaps = []
        masks = []

        for j, ann in enumerate(self.samples[idx]['anns']):
            image_shape = ann['bbox'][3], ann['bbox'][2]
            new_ann = ann.deepcopy()
            # for i in range len(ann['keypoints']//3):
            #     new_ann['keypoints'][3*i] -= new_ann['bbox'][0]
            #     new_ann['keypoints'][3*i+1] -= new_ann['bbox'][1]
            
            im = self.samples[idx]['img']
            sub_img = image.crop((ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]))

            # mask = coco_annotations_to_mask_bbox(self.samples[idx]['anns'], (im['height'], im['width']))
            #Tao mask voi shape bang voi shape cua bbox
            mask = np.ones(image_shape, dtype=np.uint8)
            mask = PIL.Image.fromarray(mask)
            
            count = self.counts[idx][j]
            peak = self.peaks[idx][j]

            # affine transformation
            shiftx = float(torch.rand(1)) * (self.random_translate[1] - self.random_translate[0]) + self.random_translate[0]
            shifty = float(torch.rand(1)) * (self.random_translate[1] - self.random_translate[0]) + self.random_translate[0]
            scale = float(torch.rand(1)) * (self.random_scale[1] - self.random_scale[0]) + self.random_scale[0]
            angle = float(torch.rand(1)) * (self.random_angle[1] - self.random_angle[0]) + self.random_angle[0]
            
            if self.keep_aspect_ratio:
                ar = float(sub_img.width) / float(sub_img.height)
                quad = get_quad(angle, (shiftx, shifty), scale, aspect_ratio=ar)
            else:
                quad = get_quad(angle, (shiftx, shifty), scale, aspect_ratio=1.0)
            
            sub_img = transform_image(sub_img, (self.image_shape[1], self.image_shape[0]), quad)
            mask = transform_image(mask, (self.target_shape[1], self.target_shape[0]), quad)
            peak = transform_peaks(count, peak, quad)
            
            count = count[None, ...]
            peak = peak[None, ...]

            stdev = float(self.stdev * self.target_shape[0])

            cmap = trt_pose.plugins.generate_cmap(counts, peaks,
                self.target_shape[0], self.target_shape[1], stdev, int(stdev * 5))

            sub_img = sub_img.convert('RGB')
            if self.transforms is not None:
                sub_img = self.transforms(sub_img)
            
            images.append(images)
            cmaps.append(cmap)
            masks.append(torch.from_numpy(np.array(mask))[None, ...])
            
        return images, cmaps, masks

    def get_part_type_counts(self):
        return torch.sum(self.counts, dim=0)
    
class CocoHumanPoseEval(object):
    
    def __init__(self, images_dir, annotation_file, image_shape, keep_aspect_ratio=False):
        
        self.images_dir = images_dir
        self.annotation_file = annotation_file
        self.image_shape = tuple(image_shape)
        self.keep_aspect_ratio = keep_aspect_ratio
        
        self.cocoGt = pycocotools.coco.COCO('annotations/person_keypoints_val2017.json')
        self.catIds = self.cocoGt.getCatIds('person')
        self.imgIds = self.cocoGt.getImgIds(catIds=self.catIds)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def evaluate(self, model, topology):
        self.parse_objects = ParseObjects(topology, cmap_threshold=0.1, link_threshold=0.1, cmap_window=5, line_integral_samples=7, max_num_parts=100, max_num_objects=100)
        
        results = []

        for n, imgId in enumerate(self.imgIds[1:]):

            # read image
            img = self.cocoGt.imgs[imgId]
            img_path = os.path.join(self.images_dir, img['file_name'])

            image = PIL.Image.open(img_path).convert('RGB')#.resize(IMAGE_SHAPE)
            
            if self.keep_aspect_ratio:
                ar = float(image.width) / float(image.height)
            else:
                ar = 1.0
                
            quad = get_quad(0.0, (0, 0), 1.0, aspect_ratio=ar)
            image = transform_image(image, self.image_shape, quad)

            data = self.transform(image).cuda()[None, ...]

            cmap, paf = model(data)
            cmap, paf = cmap.cpu(), paf.cpu()

            # object_counts, objects, peaks, int_peaks = postprocess(cmap, paf, cmap_threshold=0.05, link_threshold=0.01, window=5)
            # object_counts, objects, peaks = int(object_counts[0]), objects[0], peaks[0]

            object_counts, objects, peaks = self.parse_objects(cmap, paf)
            object_counts, objects, peaks = int(object_counts[0]), objects[0], peaks[0]

            for i in range(object_counts):
                object = objects[i]
                score = 0.0
                kps = [0]*(17*3)
                x_mean = 0
                y_mean = 0
                cnt = 0
                for j in range(17):
                    k = object[j]
                    if k >= 0:
                        peak = peaks[j][k]
                        if ar > 1.0: # w > h w/h
                            x = peak[1]
                            y = (peak[0] - 0.5) * ar + 0.5
                        else:
                            x = (peak[1] - 0.5) / ar + 0.5
                            y = peak[0]

                        x = round(float(img['width'] * x))
                        y = round(float(img['height'] * y))

                        score += 1.0
                        kps[j * 3 + 0] = x
                        kps[j * 3 + 1] = y
                        kps[j * 3 + 2] = 2
                        x_mean += x
                        y_mean += y
                        cnt += 1

                ann = {
                    'image_id': imgId,
                    'category_id': 1,
                    'keypoints': kps,
                    'score': score / 17.0
                }
                results.append(ann)
            if n % 100 == 0:
                print('%d / %d' % (n, len(self.imgIds)))


        if len(results) == 0:
            return
        
        with open('trt_pose_results.json', 'w') as f:
            json.dump(results, f)
            
        cocoDt = self.cocoGt.loadRes('trt_pose_results.json')
        cocoEval = pycocotools.cocoeval.COCOeval(self.cocoGt, cocoDt, 'keypoints')
        cocoEval.params.imgIds = self.imgIds
        cocoEval.params.catIds = [1]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()