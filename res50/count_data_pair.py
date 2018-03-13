import _init_paths
import os
os.environ['GLOG_minloglevel'] = '2'
import cv2
import math
import numpy as np
import MySQLdb
import pickle 
import caffe

def get_anno_data():
    conn = MySQLdb.connect(host='192.168.1.12', user='root', passwd='reload123', db='shampoo')
    cur = conn.cursor()
    cur.execute('''
        select o.x_pixel, o.y_pixel, o.width_pixel, o.height_pixel, o.img_id, o.type, i.url, o.id from object o inner join image i on o.img_id = i.id where o.src_src = 'citybox_test10' and o.box_status=0 and i.verify_status = 1''')
    data = cur.fetchall()

    annotations = {}
    for d in data:
        x0, y0, x1, y1 = int(d[0]), int(d[1]), int(d[0] + d[2]), int(d[1] + d[3])
        pair_id, location = int(d[6].split('_')[-2]), int(d[6].split('_')[-1].split('.')[0])

        if d[4] in annotations:
            annotations[d[4]].append({'bbox': [x0,y0,x1,y1], 'type': int(d[5]), 'center': [(x0+x1)/2, (y0+y1)/2], 'w': int(d[2]), 'h': int(d[3]), 'img_id': int(d[4]), 'pair_id': pair_id, 'location': location, 'path': d[6], 'object_id': int(d[7])})
        else:
            annotations[d[4]] = [{'bbox': [x0,y0,x1,y1], 'type': int(d[5]), 'center': [(x0+x1)/2, (y0+y1)/2], 'w': int(d[2]), 'h': int(d[3]), 'img_id': int(d[4]), 'pair_id': pair_id, 'location': location, 'path': d[6], 'object_id': int(d[7])}]

    return annotations


def test():
    data = get_anno_data()
    pair_annotations = []
    for i in range(1000):
        pair = [0,0]
        for img_id, d in data.items():
            if d[0]['pair_id'] == i:
                pair[d[0]['location'] % 2] = d
        if pair[0] and pair[1]:
            pair_annotations.append(pair)

    print len(pair_annotations)



if __name__ == '__main__':
    test()