# convert_voc_to_yolo.py

import os
import xml.etree.ElementTree as ET

# Update these paths to match your dataset
dataset_dirs = ['train', 'valid', 'test']
image_dir_base = 'images'
label_dir_base = 'labels'

# Class mapping (must match data.yaml)
class_names = [
  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
  'ا', 'ب', 'ج', 'د', 'ه', 'و', 'ز', 'ح', 'ط', 'ي', 'ك', 'ل', 'م', 'ن', 'س', 'ع', 'ف', 'ص', 'ق', 'ر', 'ش', 'ت', 'ث', 'خ', 'ذ', 'ض'
]

def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x *= dw
    w *= dw
    y *= dh
    h *= dh
    return (x, y, w, h)

def convert_annotation(xml_file, txt_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(txt_file, 'w') as f:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in class_names:
                continue
            cls_id = class_names.index(cls)
            xml_box = obj.find('bndbox')
            b = (
                float(xml_box.find('xmin').text),
                float(xml_box.find('xmax').text),
                float(xml_box.find('ymin').text),
                float(xml_box.find('ymax').text),
            )
            bb = convert_bbox((w, h), b)
            f.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")

for split in dataset_dirs:
    image_dir = os.path.join(image_dir_base, split)
    label_dir = os.path.join(label_dir_base, split)
    os.makedirs(label_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            xml_file = os.path.join(image_dir_base, split, filename.replace('.jpg', '.xml'))
            txt_file = os.path.join(label_dir, filename.replace('.jpg', '.txt'))
            if os.path.exists(xml_file):
                convert_annotation(xml_file, txt_file)
                print(f"Converted {xml_file} to {txt_file}")
            else:
                print(f"Missing XML for {filename}")