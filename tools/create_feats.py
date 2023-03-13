import os
import base64
import numpy as np
import csv
import sys
import argparse
from tqdm import tqdm
csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

def main(args):      
    count = 0
    with open(args.infeats, "rt") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in tqdm(reader):
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.decodestring(bytes(item[field], 'utf-8')),
                        dtype=np.float32).reshape((item['num_boxes'],-1))
            image_id = item['image_id']
            feats = item['features']
            np.savez_compressed(os.path.join(args.outfolder, str(image_id)), features=feats, boxes=item['boxes'])
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--infeats', default='/data1/dataset/COCO_caption/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv', help='image features')
    parser.add_argument('--outfolder', default='/data1/wlx/project2021/CVPR2022_xmodaler/open_source_dataset/mscoco_dataset/features/up_down_with_box', help='output folder')

    args = parser.parse_args()
    main(args)