import os
os.environ['GLOG_minloglevel'] = '2'
import cv2
import math
import numpy as np
import MySQLdb 

def produce_bbox(bbox, length, h, w):
    scaled_bbox = [0,0,0,0]
    scaled_bbox[0] = max(bbox[0]-length, 0)
    scaled_bbox[1] = max(bbox[1]-length, 0)
    scaled_bbox[2] = min(w, bbox[2]+length)
    scaled_bbox[3] = min(h, bbox[3]+length)
    return [int(i) for i in scaled_bbox]

conn = MySQLdb.connect(host='192.168.1.12', user='root', passwd='reload123', db='shampoo')
cur = conn.cursor()
cur.execute('''
    select o.x_pixel, o.y_pixel, o.width_pixel, o.height_pixel, o.img_id, o.type, i.url, o.id from object o inner join image i on o.img_id = i.id where o.src_src = 'citybox_test7' and o.box_status=0 and i.verify_status = 1''')
data = cur.fetchall()

annotations = {}
for d in data:
    x0, y0, x1, y1 = int(d[0]), int(d[1]), int(d[0] + d[2]), int(d[1] + d[3])
    pair_id, location = int(d[6].split('_')[-2]), int(d[6].split('_')[-1].split('.')[0])
    if d[4] in annotations:
        annotations[d[4]].append({'bbox': [x0,y0,x1,y1], 'type': int(d[5]), 'center': [(x0+x1)/2, (y0+y1)/2], 'w': int(d[2]), 'h': int(d[3]), 'img_id': int(d[4]), 'pair_id': pair_id, 'location': location, 'path': d[6], 'object_id': int(d[7])})
    else:
        annotations[d[4]] = [{'bbox': [x0,y0,x1,y1], 'type': int(d[5]), 'center': [(x0+x1)/2, (y0+y1)/2], 'w': int(d[2]), 'h': int(d[3]), 'img_id': int(d[4]), 'pair_id': pair_id, 'location': location, 'path': d[6], 'object_id': int(d[7])}]

data = annotations
pair_annotations = []
for i in range(1000):
    pair = [0,0]
    for img_id, d in data.items():
        if d[0]['pair_id'] == i:
            pair[d[0]['location'] % 2] = d
    if pair[0] and pair[1]:
        pair_annotations.append(pair)

pair_annotations = pair_annotations[0]
pair_annotations = [pair_annotations]
length = [60,90,120,150,180]
for i in range(5):
    for annotations1, annotations2 in pair_annotations:
        img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        h, w = img1.shape[:2]
        for anno1 in annotations1:
            bbox = produce_bbox(anno1['bbox'], length[i], h, w)
            img_ann1 = img1[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            cv2.imwrite('./demo/%s-%d-%d.jpg'%(anno1['object_id'], length[i], anno1['img_id']), img_ann1)
        for anno2 in annotations2:
            bbox = produce_bbox(anno2['bbox'], length[i], h, w)
            img_ann2 = img2[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            cv2.imwrite('./demo/%s-%d-%d.jpg'%(anno2['object_id'], length[i], anno2['img_id']), img_ann2)
    



