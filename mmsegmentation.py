from mimetypes import init
import os
import logging
import boto3
import io
import json
from functools import partial

import cv2
import numpy as np
import time


from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch import distributed as dist
from torchvision import transforms

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, get_single_tag_keys, DATA_UNDEFINED_NAME, is_skipped, get_local_path
from label_studio_tools.core.utils.io import get_data_dir
from label_studio_converter.brush import encode_rle, decode_rle


from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.parallel import collate

from urllib.parse import urlparse
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


image_size = 550
crop_size = 500
image_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(crop_size),
    # transforms.RandomHorizontalFlip(0.5),
    # transforms.RandomVerticalFlip(0.5),
    # transforms.ColorJitter(brightness=0.2, hue=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_cache_dir = os.path.join(os.path.dirname(__file__), 'image-cache')
os.makedirs(image_cache_dir, exist_ok=True)


def get_transformed_image(url):
    filepath = get_local_path(url)

    with open(filepath, mode='rb') as f:
        image = Image.open(f).convert('RGB')

    return image_transforms(image)

class TimelapseEmbryoDataset(Dataset):
    CLASSES = ('embryo')
    PALETTE = [[255, 0, 0]]

    def __init__(self,
                 image_urls,
                 gt_seg_maps):
        self.image_urls = image_urls
        self.images = []
        for image_url in self.image_urls:
            image = get_transformed_image(image_url)
            self.images.append(image)
        self.gt_seg_maps = gt_seg_maps
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return {'img': self.images[index], 'img_metas': {}, 'gt_semantic_seg': np.expand_dims(self.gt_seg_maps[index], 0)}


class MMSegmentation(LabelStudioMLBase):
    def __init__(self, config_file='mmsegmentation/fastfcn/fastfcn_r50-d32_timelapse_embryo.py',
                 checkpoint_file='mmsegmentation/fastfcn_r50-d32_jpu_aspp_512x512_80k_ade20k_20211013_190619-3aa40f2d.pth',
                 image_dir=None,
                 labels_file=None, device='cuda', num_epochs=5, allow_all_task=False, **kwargs):        
        super().__init__(**kwargs)
        config_file = config_file or os.environ['config_file']
        checkpoint_file = checkpoint_file or os.environ['checkpoint_file']
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.labels_file = labels_file
        self.num_epochs = num_epochs
        # default Label Studio image upload folder
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        self.image_dir = image_dir or upload_dir
        logger.debug(f'{self.__class__.__name__} reads images from {self.image_dir}')
        if self.labels_file and os.path.exists(self.labels_file):
            self.label_map = json_load(self.labels_file)
        else:
            self.label_map = {}

        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'BrushLabels', 'Image')
        schema = list(self.parsed_label_config.values())[0]
        self.labels_in_config = set(self.labels_in_config)

        # Collect label maps from `predicted_values="airplane,car"` attribute in <Label> tag
        self.labels_attrs = schema.get('labels_attrs')
        if self.labels_attrs:
            for label_name, label_attrs in self.labels_attrs.items():
                for predicted_value in label_attrs.get('predicted_values', '').split(','):
                    self.label_map[predicted_value] = label_name

        print('Load new model from: ', config_file, checkpoint_file)
        self.model = init_segmentor(config_file, checkpoint_file, device=device)
        self.device = device
        self.id2label = {1: 'embryo'}
        self.label2id = {v:k for k, v in self.id2label.items()}
        self.palette = {1: [255, 0, 0]}
        self.allow_all_task = allow_all_task
        
    def _get_image_url(self, task):
        image_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        if image_url.startswith('s3://'):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3')
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': key}
                )
            except ClientError as exc:
                logger.warning(f'Can\'t generate presigned URL for {image_url}. Reason: {exc}')
        return image_url

    def predict(self, tasks, **kwargs):
        if len(tasks) > 1 and not self.allow_all_task:
            raise RuntimeError('all task prediction not allowed')
        predictions = []
        for task in tasks:
            image_url = self._get_image_url(task)
            image_path = self.get_local_path(image_url)
            img_width, img_height = get_image_size(image_path)
            model_result = inference_segmentor(self.model, image_path)[0]
            results = []
            unique_labels = np.unique(model_result)
            avg_score = 0
            total = 0
            for label in unique_labels:
                if label == 0:
                    continue
                binary_mask = (model_result == label) # get binary mask of current label
                mask = np.zeros((img_height, img_width, 4), dtype=int)
                mask[binary_mask == True] = self.palette[label] + [255]
                rle_strs = encode_rle(mask.flatten())

                score = binary_mask.sum() / (img_width*img_height) # 愈少愈低分
                results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'brushlabels',
                    "original_width": img_width,
                    "original_height": img_height,
                    "image_rotation": 0,
                    'value': {
                        "brushlabels": [self.id2label[label]],
                        "format": "rle",
                        "rle": rle_strs
                    },
                    'score': score
                })
                avg_score += score
                total += 1
            predictions.append({'result': results, 'score': avg_score/total})
        return predictions

    def fit(self, completions, workdir=None, batch_size=8, **kwargs):
        
        image_urls, gt_seg_maps = [], []
        print('Collecting annotations...')
        for i, completion in enumerate(completions):
            if is_skipped(completion):
                continue
            
            results = completion['annotations'][0]['result']
            if len(results) == 0:
                # no embryo
                gt_seg_map = np.zeros((500, 500), dtype=np.uint8)    
            else:
                width = results[0]['original_width']
                height = results[0]['original_height']
                gt_seg_map = np.zeros((height, width), dtype=np.uint8)
                for i, result in enumerate(results):
                    rle = result['value']['rle']
                    label = '-'.join(result['value']['brushlabels'])
                    image = decode_rle(rle)
                    image = np.reshape(image, [height, width, 4])[:,:,3] # 4 channel is all the same
                    gt_seg_map[image > 127.5] = self.label2id[label]
            
            image_urls.append(completion['data'][self.value])
            gt_seg_maps.append(gt_seg_map)
        
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '5678'
            dist.init_process_group(backend='nccl', rank=0, world_size=1)
        print(f'Creating dataset with {len(image_urls)} images...')
        dataset = TimelapseEmbryoDataset(image_urls, gt_seg_maps)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=partial(collate, samples_per_gpu=batch_size))
        print('Train model...')
        optimizer = optim.SGD(self.model.parameters(), lr=0.001)
        self.model = train(self.model, dataloader, optimizer, num_epochs=self.num_epochs, device=self.device)
        print('Save model...')
        # TODO model not updated hack
        # model_path = os.path.join(workdir, 'model.pt')
        model_path = self.checkpoint_file
        torch.save({'meta': {'CLASSES': ('embryo'), 'PALETTE': [[255, 0, 0]]}, 'state_dict': self.model.state_dict()}, model_path)
        return {'model_path': model_path, 'classes': ['embryos']}
    
def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data
        
def train(model, dataloader, optimizer, num_epochs=10, device='cuda'):
    since = time.time()

    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0
        # Iterate over data.
        for data_batch in dataloader:            
            data_batch['img'] = data_batch['img'].to(device)
            data_batch['gt_semantic_seg'] = data_batch['gt_semantic_seg'].to(device).long()
            optimizer.zero_grad()
            outputs = model.train_step(data_batch, optimizer)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            # statistics
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(dataloader)

        print('Train Loss: {}'.format(epoch_loss))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model