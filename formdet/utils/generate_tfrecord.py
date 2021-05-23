import tensorflow as tf

# from object_detection.utils import dataset_util

import os
import glob
import pandas as pd
import io

import json

from pathlib import Path
from PIL import Image

from object_detection.utils import dataset_util, label_map_util
from collections import Counter

def files_iter(dataset_root: Path, pattern: str = '**/*', allowed_extensions = ['.json', '.png']):
    for file in dataset_root.rglob(pattern):
        if file.suffix in allowed_extensions:
            yield file


def dataset_description_iter(dataset_root: Path):
    # for png_file in files_iter(dataset_root = dataset_root, allowed_extensions=['.png']):
    #     stem = png_file.stem
    #     dir = png_file.parent
    #     yield dir / f'{stem}.json', png_file
    for json_file in files_iter(dataset_root = dataset_root, allowed_extensions=['.json']):
        stem = json_file.stem
        dir = json_file.parent
        yield json_file, dir / f'{stem}.png'

class FormDetReader:

    def __init__(self, dataset_root, coeff = 300./96., exclude_cats=None, default_classes_dict=None) -> None:
        super().__init__()

        if default_classes_dict is None:
            default_classes_dict = dict()
        if exclude_cats is None:
            exclude_cats = {'image', 'dropdown'}

        self.dataset_root = dataset_root
        self.classes_dict = default_classes_dict
        self.coeff = coeff
        self.stat_max_fields_in_file = 0
        self.category_stat = Counter()
        
        self.exclude_cats = exclude_cats

    def create_index(self, tf_filename):
        with tf.io.TFRecordWriter(str(tf_filename)) as writer:
            for json_file, image_file in dataset_description_iter(self.dataset_root):
                example_proto = self.create_tf_example(json_file, image_file)
                writer.write(example_proto.SerializeToString())

    def create_tf_example(self, json_file, image_file):
        if json_file.exists():
            with json_file.open(mode='r') as json_fid:
                data = json.load(json_fid)
        else:
            data = dict()
            data['fields'] = []

        filename = str(image_file)
        with tf.io.gfile.GFile(filename, 'rb') as fid:
            encoded_png = fid.read()
        encoded_png_io = io.BytesIO(encoded_png)
        image = Image.open(encoded_png_io)
        width, height = image.size
        image_format = b'png'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []
        
        filename = filename.encode('utf8')
        
        used_fields = 0
        for field in data['fields']:
            #     print(f"bbox: {field['bbox']} type: {field['type']}")
            bbox = field['bbox']
            x = bbox['x'] * self.coeff
            y = bbox['y'] * self.coeff
            w = bbox['width'] * self.coeff
            h = bbox['height'] * self.coeff

            ctext = field['type']
            
            if ctext.startswith('checkmark_'):
                ctext = 'checkmark'
            
            if ctext in self.exclude_cats:
                continue

            self.category_stat.update([ctext])
            used_fields += 1
            ctext = ctext.encode('utf8')

            id = len(self.classes_dict)+1
            if ctext in self.classes_dict:
                id = self.classes_dict[ctext]
            else:
                self.classes_dict[ctext] = id

            xmins.append(x / width)
            xmaxs.append((x + w) / width)
            ymins.append(y / height)
            ymaxs.append((y + h) / height)
            classes_text.append(ctext)
            classes.append(id)
            
        self.stat_max_fields_in_file = max(self.stat_max_fields_in_file, used_fields)
            
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_png),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example
    
    def create_label_map(self, label_map_file: Path):
        with label_map_file.open(mode='w') as label_map_writer:
            for k, v in self.classes_dict.items():
                label_map_writer.write((f"item {{\n\tid:{v}\n\tname: '{k.decode('utf8')}'\n}}\n"))


if __name__=='__main__':
    for json_file, image_file in dataset_description_iter(Path.cwd() / 'data'):
        print(f'{json_file} -- {image_file}')
