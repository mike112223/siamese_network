import _init_paths
import os
os.environ['GLOG_minloglevel'] = '2'
import cv2
import math
import numpy as np
import MySQLdb
import pickle 
import caffe

def IOU(Reframe, GTframe):
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]-Reframe[0]
    height1 = Reframe[3]-Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]-GTframe[0]
    height2 = GTframe[3]-GTframe[1]

    endx = max(x1+width1,x2+width2)
    startx = min(x1,x2)
    width = width1+width2-(endx-startx)

    endy = max(y1+height1,y2+height2)
    starty = min(y1,y2)
    height = height1+height2-(endy-starty)

    if width <=0 or height <= 0:
        ratio = 0
    else:
        Area = width*height
        Area1 = width1*height1 
        Area2 = width2*height2
        ratio = Area*1./(Area1+Area2-Area)
    return ratio



def merge_by_two_annotations_test(annotations1, annotations2, img1, img2):
    h, w = img1.shape[0:2]
    color_type = {i: (np.random.rand(3) * 256).astype(int) for i in range(10000)}

    img = np.zeros((h*2, w*2, 3))  

    distinct_annotations1, distinct_annotations2, duplicate_pair_annos = merge_by_two_annotations(annotations1, annotations2, img1, img2, True)
    for anno in annotations1:
        x0,y0,x1,y1 = anno['bbox']
        cv2.rectangle(img1, (x0,y0), (x1,y1), color_type[anno['type']], 3)
        cv2.putText(img1, "%d, %d" %(anno['center'][0], anno['center'][1]), (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_type[anno['type']], 4)

    for anno in annotations2:
        x0,y0,x1,y1 = anno['bbox']
        cv2.rectangle(img2, (x0,y0), (x1,y1), color_type[anno['type']], 3)
        cv2.putText(img2, "%d, %d" %(anno['center'][0], anno['center'][1]), (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_type[anno['type']], 4)

    img[:h,:w] = img1 
    img[h:,w:] = img2

    for anno1, anno2 in duplicate_pair_annos:
        x0,y0,x1,y1 = anno1['bbox']
        x2,y2,x3,y3 = anno2['bbox']
        cv2.line(img, (anno1['center'][0], anno1['center'][1]), (anno2['center'][0] + w, anno2['center'][1] + h), (np.random.rand(3) * 256).astype(int), 2)
        #cv2.putText(img, "%d, %d" %(anno1['center'][0], anno1['center'][1]), (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_type[anno1['type']], 4)
        #cv2.putText(img, "%d, %d" %(anno2['center'][0], anno2['center'][1]), (x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_type[anno1['type']], 4)


    test_img = np.zeros((h*2, w*2, 3))
    merge_img = np.zeros((h, w, 3))

    for anno in distinct_annotations1:
        x0,y0,x1,y1 = anno['bbox']
        test_img[y0:y1, x0:x1] = img1[y0:y1, x0:x1]
    
    for anno in distinct_annotations2:
        x0,y0,x1,y1 = anno['bbox']
        test_img[y0+h:y1+h, x0+w:x1+w] = img2[y0:y1, x0:x1]

    cv2.imwrite('./result/%s.jpg' % annotations1[0]['pair_id'], img)
    cv2.imwrite('./result/%s_test.jpg' % annotations1[0]['pair_id'], test_img)

    return distinct_annotations1, distinct_annotations2


def merge_by_two_annotations(annotations1, annotations2, image1, image2, require_pairs = False):

    h, w = image1.shape[0:2]


    for anno in annotations1:
        x0,y0,x1,y1 = anno['bbox']
        anno['center'] = [(x0+x1)/2, (y0+y1)/2]
        anno['w'] = x1 - x0
        anno['h'] = y1 - y0
        anno['score'] = anno.get('score', 1.0)

    for anno in annotations2:
        x0,y0,x1,y1 = anno['bbox']
        anno['center'] = [(x0+x1)/2, (y0+y1)/2]
        anno['w'] = x1 - x0
        anno['h'] = y1 - y0
        anno['score'] = anno.get('score', 1.0)

    distinct_annotations1, inter_annotations1 = [], [i for i in annotations1 if i['w'] > 0 and i['h'] >0]
    distinct_annotations2, inter_annotations2 = [], [i for i in annotations2 if i['w'] > 0 and i['h'] >0] 
    
    anno_scaled_images = []
    for anno in inter_annotations1:
        #anno_scaled_bbox = scale_bbox(anno['bbox'], 2.5, min_len=120)
        anno_scaled_bbox = produce_bbox(anno['bbox'], 120, h, w)
        x1, y1, x2, y2 = anno_scaled_bbox
        anno_scaled_image = image1[y1: y2, x1: x2]

        #
        cv2.imwrite('./demo_merge/%s-%d.jpg'%(anno['object_id'], anno['img_id']), anno_scaled_image)

        anno_scaled_images.append(anno_scaled_image)

    anno_scaled_images_feature = get_siamese_feature(anno_scaled_images)
    for i in range(len(inter_annotations1)):
        inter_annotations1[i]['siamese_feature'] = anno_scaled_images_feature[i]
    
    anno_scaled_images = []
    for anno in inter_annotations2:
        #anno_scaled_bbox = scale_bbox(anno['bbox'], 2.5, min_len=120)
        anno_scaled_bbox = produce_bbox(anno['bbox'], 120, h, w)
        x1, y1, x2, y2 = anno_scaled_bbox
        anno_scaled_image = image2[y1: y2, x1: x2]

        #
        cv2.imwrite('./demo_merge/%s-%d.jpg'%(anno['object_id'], anno['img_id']), anno_scaled_image)

        anno_scaled_images.append(anno_scaled_image)

    anno_scaled_images_feature = get_siamese_feature(anno_scaled_images)
    for i in range(len(inter_annotations2)):
        inter_annotations2[i]['siamese_feature'] = anno_scaled_images_feature[i]


    #dist_log = open('dist_log.txt','w')
    len1, len2 = len(inter_annotations1), len(inter_annotations2)
    dist_map = np.zeros((len1, len2))
    for i in range(len1):
        for j in range(len2):
            anno1, anno2 = inter_annotations1[i], inter_annotations2[j]
            dist_map[i,j] = np.linalg.norm(anno1['siamese_feature'] - anno2['siamese_feature'])
            if anno1['type'] != anno2['type']:
                #ratio1 = 1.0*anno1['h']/anno1['w']
                #ratio2 = 1.0*anno2['h']/anno2['w']
                #dist_map[i, j] *= 1.0*max(ratio1, ratio2)/min(ratio1, ratio2)
                dist_map[i,j] *= 2.0
            #text = '%d, %d'%(anno1['center'][0], anno1['center'][1])+'-'+'%d, %d'%(anno2['center'][0], anno2['center'][1])+ '------' + '%f \r\n'%(dist_map[i,j])
            #dist_log.writelines(text)
    #dist_log.writelines('--------------------------')
    #dist_log.close()  
    
    paired_index1 = []
    paired_index2 = []
    duplicate_pair_annos_index = []
    
    while True:
        if dist_map.min() > 1.0:
            break
        min_index = np.where(dist_map==dist_map.min())
        i, j = min_index[0][0], min_index[1][0]
        dist_map[i], dist_map[:,j] = 100, 100

        pair_anno1, pair_anno2 = inter_annotations1[i], inter_annotations2[j]
        for x in range(len1):
            for y in range(len2):
                if dist_map[x,y] > 1.0:
                    continue
                anno1, anno2 = inter_annotations1[x], inter_annotations2[y]
                if ((anno1['bbox'][0] > pair_anno1['bbox'][2] + 10) and (anno2['bbox'][2] < pair_anno2['bbox'][0] - 10)) or \
                ((anno2['bbox'][0] > pair_anno2['bbox'][2] + 10) and (anno1['bbox'][2] < pair_anno1['bbox'][0] - 10)) or \
                ((anno1['bbox'][1] > pair_anno1['bbox'][3] + 10) and (anno2['bbox'][3] < pair_anno2['bbox'][1] - 10)) or \
                ((anno2['bbox'][1] > pair_anno2['bbox'][3] + 10) and (anno1['bbox'][3] < pair_anno1['bbox'][1] - 10)) :
                    dist_map[x, y] = 100

        duplicate_pair_annos_index.append([i, j])
        paired_index1.append(i)
        paired_index2.append(j)

    for i in range(len1):
        if i not in paired_index1:
            distinct_annotations1.append(inter_annotations1[i])

    for i in range(len2):
        if i not in paired_index2:
            distinct_annotations2.append(inter_annotations2[i])
    
    for anno in inter_annotations1 + inter_annotations2:
        anno.pop('siamese_feature')


    duplicate_pair_annos = [[inter_annotations1[i[0]], inter_annotations2[i[1]]] for i in duplicate_pair_annos_index]
    for anno1, anno2 in duplicate_pair_annos:
        if anno1['type'] != anno2['type']:
            anno1['type'], anno2['type'] = [anno2['type'], anno2['type']] if anno2['score'] > anno1['score'] else [anno1['type'], anno1['type']]
        if anno1['center'][0] - w * 0.26 < 0 and anno1 not in distinct_annotations1:
            distinct_annotations1.append(anno1)
        elif anno2 not in distinct_annotations2:
            distinct_annotations2.append(anno2)
    
    if not require_pairs:
        return distinct_annotations1, distinct_annotations2
    else:
        return distinct_annotations1, distinct_annotations2, duplicate_pair_annos

def _get_transformer(shape, mean, lib='cf'):
    transformer = caffe.io.Transformer({'data':shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', mean)

    if lib == 'cf':
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2,1,0))

    return transformer

net = None
transformer = None
batch_size = 8
def init_siamese_net():
    global net, transformer, batch_size
    if not net or not transformer:
        caffe.set_mode_gpu()
        caffe.set_device(1)

        model_def = 'merge_models/deploy.prototxt'
        model_weights = 'models/resnet50_v0_iter_200000.caffemodel'
        net = caffe.Net(model_def, model_weights, caffe.TEST)
        net.blobs['data'].reshape(batch_size, 3, 224, 224)
        transformer = _get_transformer(net.blobs['data'].data.shape, np.array([104, 117, 123]), 'cv')

def get_siamese_feature(images):
    global batch_size
    features = []
    while len(images):
        if len(images) > batch_size:
            net.blobs['data'].data[...] = [transformer.preprocess('data', images.pop(0)) for i in range(batch_size)] 
            net.forward()
            output = net.blobs['feat'].data.copy()
            features.extend([output[i] for i in range(output.shape[0])])
        else:
            net.blobs['data'].reshape(len(images), 3, 224, 224)
            net.blobs['data'].data[...] = [transformer.preprocess('data', images.pop(0)) for i in range(len(images))] 
            net.forward()
            net.blobs['data'].reshape(batch_size, 3, 224, 224)
            output = net.blobs['feat'].data.copy()
            features.extend([output[i] for i in range(output.shape[0])])
        
    return features

def scale_bbox(bbox, scale_size, min_len=None, max_len=None):
    """
    Argus:
        -bbox: (x1, y1, x2, y2), shape = (1, 4)
    """
    scaled_bbox = [0,0,0,0]
    widths = bbox[2] - bbox[0] + 1.0
    heights = bbox[3] - bbox[1] + 1.0
    ctr_x = bbox[0] + 0.5 * widths
    ctr_y = bbox[1] + 0.5 * heights
    
    if min_len:
        scale_size = max(scale_size, min_len/widths, min_len/heights)
    if max_len:
        scale_size = min(scale_size, max_len/widths, max_len/heights)

    scaled_widths = widths * scale_size
    scaled_heights = heights * scale_size

    scaled_bbox[0] = max(0, ctr_x - 0.5 * scaled_widths)
    scaled_bbox[2] = ctr_x + 0.5 * scaled_widths
    scaled_bbox[1] = max(0, ctr_y - 0.5 * scaled_heights)
    scaled_bbox[3] = ctr_y + 0.5 * scaled_heights

    return [int(i) for i in scaled_bbox]

def produce_bbox(bbox, length, h, w):

    scaled_bbox = [0,0,0,0]
    scaled_bbox[0] = max(bbox[0]-length, 0)
    scaled_bbox[1] = max(bbox[1]-length, 0)
    scaled_bbox[2] = min(w, bbox[2]+length)
    scaled_bbox[3] = min(h, bbox[3]+length)

    return [int(i) for i in scaled_bbox]   

def get_anno_data():
    conn = MySQLdb.connect(host='192.168.1.12', user='root', passwd='reload123', db='shampoo')
    cur = conn.cursor()
    cur.execute('''
        select o.x_pixel, o.y_pixel, o.width_pixel, o.height_pixel, o.img_id, o.type, i.url, o.id from object o inner join image i on o.img_id = i.id where o.src_src = 'shelf_overlook_test9' and o.box_status=0 and i.verify_status = 1''')
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

def get_anno_data1():
    conn = MySQLdb.connect(host='192.168.1.12', user='root', passwd='reload123', db='shampoo')
    cur = conn.cursor()
    cur.execute('''
        select o.x_pixel, o.y_pixel, o.width_pixel, o.height_pixel, o.img_id, o.type, i.url, o.id from object o inner join image i on o.img_id = i.id where o.src_src = 'shelf_overlook_test10' and o.box_status=0 and i.verify_status = 1''')
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

def get_anno_data2():
    conn = MySQLdb.connect(host='192.168.1.12', user='root', passwd='reload123', db='shampoo')
    cur = conn.cursor()
    cur.execute('''
        select o.x_pixel, o.y_pixel, o.width_pixel, o.height_pixel, o.img_id, o.type, i.url, o.id from object o inner join image i on o.img_id = i.id where o.src_src = 'shelf_overlook_test11' and o.box_status=0 and i.verify_status = 1''')
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

def get_anno_data3():
    conn = MySQLdb.connect(host='192.168.1.12', user='root', passwd='reload123', db='shampoo')
    cur = conn.cursor()
    cur.execute('''
        select o.x_pixel, o.y_pixel, o.width_pixel, o.height_pixel, o.img_id, o.type, i.url, o.id from object o inner join image i on o.img_id = i.id where o.src_src = 'citybox_test6' and o.box_status=0 and i.verify_status = 1''')
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

def get_anno_data4():
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

    return annotations

def get_anno_data5():
    conn = MySQLdb.connect(host='192.168.1.12', user='root', passwd='reload123', db='shampoo')
    cur = conn.cursor()
    cur.execute('''
        select o.x_pixel, o.y_pixel, o.width_pixel, o.height_pixel, o.img_id, o.type, i.url, o.id from object o inner join image i on o.img_id = i.id where o.src_src = 'citybox_test8' and o.box_status=0 and i.verify_status = 1''')
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

def get_anno_data6():
    conn = MySQLdb.connect(host='192.168.1.12', user='root', passwd='reload123', db='shampoo')
    cur = conn.cursor()
    cur.execute('''
        select o.x_pixel, o.y_pixel, o.width_pixel, o.height_pixel, o.img_id, o.type, i.url, o.id from object o inner join image i on o.img_id = i.id where o.src_src = 'citybox_test8_2' and o.box_status=0 and i.verify_status = 1''')
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

def get_anno_data7():
    conn = MySQLdb.connect(host='192.168.1.12', user='root', passwd='reload123', db='shampoo')
    cur = conn.cursor()
    cur.execute('''
        select o.x_pixel, o.y_pixel, o.width_pixel, o.height_pixel, o.img_id, o.type, i.url, o.id from object o inner join image i on o.img_id = i.id where o.src_src = 'citybox_test9' and o.box_status=0 and i.verify_status = 1''')
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

def test1():
    data = get_anno_data()
    pair_annotations = []
    for i in range(1000):
        pair = [0,0]
        for img_id, d in data.items():
            if d[0]['pair_id'] == i:
                pair[d[0]['location'] % 2] = d
        if pair[0] and pair[1]:
            pair_annotations.append(pair)

    conn = MySQLdb.connect(host='192.168.1.12', user='root', passwd='reload123', db='shampoo')
    cur = conn.cursor()
    cur.execute('select object_id, same_object_id from object_sames_9 where status = 1')
    data = cur.fetchall()
    object_id_pair = [int(i[0]) for i in data] + [int(i[1]) for i in data]
    
    cur.execute('select distinct(a.img_id) from object_sames_9 os inner join object a on os.object_id =a.id inner join object b on os.same_object_id =b.id where a.type<>b.type and os.status = 1;')
    data = cur.fetchall()
    bug_img_ids = [int(i[0]) for i in data]
    
    import time
    start = time.time()
    result = []
    wrong = []

    for annotations1, annotations2 in pair_annotations:
        bad_case = [18, 587, 850]
        # if annotations1[0]['pair_id'] not in bad_case:
        #     continue



        print annotations1[0]['pair_id']
        
        # right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        # img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        # a, b = merge_by_two_annotations(annotations1, annotations2, img1, img2, False)
        # result.append(int(right_count == len(a) + len(b)))
        
        
        right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        a, b = merge_by_two_annotations(annotations1, annotations2, img1, img2)
        result.append(int(right_count == len(a) + len(b)))


        if right_count != len(a) + len(b):
            wrong.append(annotations1[0]['pair_id'])
        print 'right_count: ', right_count, 'predict_count: ', len(a) + len(b), 'acc: ', 1.0 * sum(result) / len(result), 'wrong: ', wrong
        
def test2():
    import pickle
    with open('merge_1.pickle') as f:
        annotations1, annotations2, img1, img2 = pickle.load(f)
    a, b = merge_by_two_annotations_test(annotations1, annotations2, img1, img2)

def test3():
    data = get_anno_data1()
    pair_annotations = []
    for i in range(1000):
        pair = [0,0]
        for img_id, d in data.items():
            if d[0]['pair_id'] == i:
                pair[d[0]['location'] % 2] = d
        if pair[0] and pair[1]:
            pair_annotations.append(pair)

    conn = MySQLdb.connect(host='192.168.1.12', user='root', passwd='reload123', db='shampoo')
    cur = conn.cursor()
    cur.execute('select object_id, same_object_id from object_sames_10 where status = 1')
    data = cur.fetchall()
    object_id_pair = [int(i[0]) for i in data] + [int(i[1]) for i in data]
    
    cur.execute('select distinct(a.img_id) from object_sames_10 os inner join object a on os.object_id =a.id inner join object b on os.same_object_id =b.id where a.type<>b.type and os.status = 1;')
    data = cur.fetchall()
    bug_img_ids = [int(i[0]) for i in data]
    
    import time
    start = time.time()
    result = []
    wrong = []
    for annotations1, annotations2 in pair_annotations:
        bad_case = [322, 703]
        # if annotations1[0]['pair_id'] not in bad_case:
        #     continue
        print annotations1[0]['pair_id']    
        
        right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        a, b = merge_by_two_annotations(annotations1, annotations2, img1, img2, False)
        result.append(int(right_count == len(a) + len(b)))
        
        # right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        # img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        # a, b = merge_by_two_annotations_test(annotations1, annotations2, img1, img2)
        # result.append(int(right_count == len(a) + len(b)))


        if right_count != len(a) + len(b):
            wrong.append(annotations1[0]['pair_id'])
        print 'right_count: ', right_count, 'predict_count: ', len(a) + len(b), 'acc: ', 1.0 * sum(result) / len(result), 'wrong: ', wrong

def test4():
    data = get_anno_data2()
    pair_annotations = []
    for i in range(1000):
        pair = [0,0]
        for img_id, d in data.items():
            if d[0]['pair_id'] == i:
                pair[d[0]['location'] % 2] = d
        if pair[0] and pair[1]:
            pair_annotations.append(pair)

    conn = MySQLdb.connect(host='192.168.1.12', user='root', passwd='reload123', db='shampoo')
    cur = conn.cursor()
    cur.execute('select object_id, same_object_id from object_sames_11 where status = 1')
    data = cur.fetchall()
    object_id_pair = [int(i[0]) for i in data] + [int(i[1]) for i in data]
    
    cur.execute('select distinct(a.img_id) from object_sames_11 os inner join object a on os.object_id =a.id inner join object b on os.same_object_id =b.id where a.type<>b.type and os.status = 1;')
    data = cur.fetchall()
    bug_img_ids = [int(i[0]) for i in data]
    
    result = []
    wrong = []
    for annotations1, annotations2 in pair_annotations:
        bad_case = [82, 126, 262, 367]
        #if annotations1[0]['pair_id'] not in bad_case:
        #    continue
        print annotations1[0]['pair_id']
        
        # right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        # img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        # a, b = merge_by_two_annotations(annotations1, annotations2, img1, img2, False)
        # result.append(int(right_count == len(a) + len(b)))
        
        right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        a, b = merge_by_two_annotations(annotations1, annotations2, img1, img2)
        result.append(int(right_count == len(a) + len(b)))


        if right_count != len(a) + len(b):
            wrong.append(annotations1[0]['pair_id'])
        print 'right_count: ', right_count, 'predict_count: ', len(a) + len(b), 'acc: ', 1.0 * sum(result) / len(result), 'wrong: ', wrong

def test5():
    import json
    with open('debug.json') as f:
        debug = json.loads(f.read())
    with open('debug_output_v5.json') as f:
        debug_output = json.loads(f.read())
    for i in range(6):
        for cam_id in ['a', 'b']:
            print cam_id, i
            img1, img2 = cv2.imread(debug[i]['%s_cam_0' % cam_id]), cv2.imread(debug[i]['%s_cam_1' % cam_id])     
            annotations1, annotations2 = debug_output[i][u'%s_info' % cam_id][0:2]
            for a in annotations1 + annotations2:
                a['pair_id'] = cam_id + str(i)
            a, b = merge_by_two_annotations_test(annotations1, annotations2, img1, img2)

def test6():
    data = get_anno_data3()
    pair_annotations = []
    for i in range(1000):
        pair = [0,0]
        for img_id, d in data.items():
            if d[0]['pair_id'] == i:
                pair[d[0]['location'] % 2] = d
        if pair[0] and pair[1]:
            pair_annotations.append(pair)

    conn = MySQLdb.connect(host='192.168.1.12', user='root', passwd='reload123', db='shampoo')
    cur = conn.cursor()
    cur.execute('select object_id, same_object_id from object_sames_citybox_test6 where status = 1')
    data = cur.fetchall()
    object_id_pair = [int(i[0]) for i in data] + [int(i[1]) for i in data]
    
    cur.execute('select distinct(a.img_id) from object_sames_citybox_test6 os inner join object a on os.object_id =a.id inner join object b on os.same_object_id =b.id where a.type<>b.type and os.status = 1;')
    data = cur.fetchall()
    bug_img_ids = [int(i[0]) for i in data]
    
    result = []
    wrong = []
    for annotations1, annotations2 in pair_annotations:
        bad_case = [354] # [319, 332, 336, 345, 354]
        if annotations1[0]['pair_id'] not in bad_case:
            continue
        #print annotations1[0]['pair_id']
        #print annotations1[0]
        #print annotations2[0]
        
        # right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        # img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        # a, b = merge_by_two_annotations(annotations1, annotations2, img1, img2, False)
        # result.append(int(right_count == len(a) + len(b)))
        
        right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        a, b = merge_by_two_annotations(annotations1, annotations2, img1, img2)
        result.append(int(right_count == len(a) + len(b)))


        if right_count != len(a) + len(b):
            wrong.append(annotations1[0]['pair_id'])
        print 'right_count: ', right_count, 'predict_count: ', len(a) + len(b), 'acc: ', 1.0 * sum(result) / len(result), 'wrong: ', wrong


def test7():
    data = get_anno_data4()
    pair_annotations = []
    for i in range(1000):
        pair = [0,0]
        for img_id, d in data.items():
            if d[0]['pair_id'] == i:
                pair[d[0]['location'] % 2] = d
        if pair[0] and pair[1]:
            pair_annotations.append(pair)

    conn = MySQLdb.connect(host='192.168.1.12', user='root', passwd='reload123', db='shampoo')
    cur = conn.cursor()
    cur.execute('select object_id, same_object_id from object_sames_citybox_test7 where status = 1')
    data = cur.fetchall()
    object_id_pair = [int(i[0]) for i in data] + [int(i[1]) for i in data]
    
    cur.execute('select distinct(a.img_id) from object_sames_citybox_test7 os inner join object a on os.object_id =a.id inner join object b on os.same_object_id =b.id where a.type<>b.type and os.status = 1;')
    data = cur.fetchall()
    bug_img_ids = [int(i[0]) for i in data]
    
    result = []
    wrong = []
    for annotations1, annotations2 in pair_annotations:
        bad_case = [405, 416]
        #if annotations1[0]['pair_id'] not in bad_case:
        #    continue
        print annotations1[0]['pair_id']
        
        # right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        # img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        # a, b = merge_by_two_annotations(annotations1, annotations2, img1, img2, False)
        # result.append(int(right_count == len(a) + len(b)))
        
        right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        a, b = merge_by_two_annotations(annotations1, annotations2, img1, img2)
        result.append(int(right_count == len(a) + len(b)))


        if right_count != len(a) + len(b):
            wrong.append(annotations1[0]['pair_id'])
        print 'right_count: ', right_count, 'predict_count: ', len(a) + len(b), 'acc: ', 1.0 * sum(result) / len(result), 'wrong: ', wrong

def test8():
    data = get_anno_data5()
    pair_annotations = []
    for i in range(1000):
        pair = [0,0]
        for img_id, d in data.items():
            if d[0]['pair_id'] == i:
                pair[d[0]['location'] % 2] = d
        if pair[0] and pair[1]:
            pair_annotations.append(pair)

    conn = MySQLdb.connect(host='192.168.1.12', user='root', passwd='reload123', db='shampoo')
    cur = conn.cursor()
    cur.execute('select object_id, same_object_id from object_sames_citybox_test8 where status = 1')
    data = cur.fetchall()
    object_id_pair = [int(i[0]) for i in data] + [int(i[1]) for i in data]
    
    cur.execute('select distinct(a.img_id) from object_sames_citybox_test8 os inner join object a on os.object_id =a.id inner join object b on os.same_object_id =b.id where a.type<>b.type and os.status = 1;')
    data = cur.fetchall()
    bug_img_ids = [int(i[0]) for i in data]
    
    result = []
    wrong = []
    for annotations1, annotations2 in pair_annotations:
        bad_case = [82, 126, 262, 367]
        #if annotations1[0]['pair_id'] not in bad_case:
        #    continue
        print annotations1[0]['pair_id']
        
        # right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        # img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        # a, b = merge_by_two_annotations(annotations1, annotations2, img1, img2, False)
        # result.append(int(right_count == len(a) + len(b)))
        
        right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        a, b = merge_by_two_annotations(annotations1, annotations2, img1, img2)
        result.append(int(right_count == len(a) + len(b)))


        if right_count != len(a) + len(b):
            wrong.append(annotations1[0]['pair_id'])
        print 'right_count: ', right_count, 'predict_count: ', len(a) + len(b), 'acc: ', 1.0 * sum(result) / len(result), 'wrong: ', wrong

def test9():
    data = get_anno_data6()
    pair_annotations = []
    for i in range(1000):
        pair = [0,0]
        for img_id, d in data.items():
            if d[0]['pair_id'] == i:
                pair[d[0]['location'] % 2] = d
        if pair[0] and pair[1]:
            pair_annotations.append(pair)

    conn = MySQLdb.connect(host='192.168.1.12', user='root', passwd='reload123', db='shampoo')
    cur = conn.cursor()
    cur.execute('select object_id, same_object_id from object_sames_citybox_test8_2 where status = 1')
    data = cur.fetchall()
    object_id_pair = [int(i[0]) for i in data] + [int(i[1]) for i in data]
    
    cur.execute('select distinct(a.img_id) from object_sames_citybox_test8_2 os inner join object a on os.object_id =a.id inner join object b on os.same_object_id =b.id where a.type<>b.type and os.status = 1;')
    data = cur.fetchall()
    bug_img_ids = [int(i[0]) for i in data]
    
    result = []
    wrong = []
    for annotations1, annotations2 in pair_annotations:
        bad_case = [82, 126, 262, 367]
        #if annotations1[0]['pair_id'] not in bad_case:
        #    continue
        print annotations1[0]['pair_id']
        
        # right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        # img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        # a, b = merge_by_two_annotations(annotations1, annotations2, img1, img2, False)
        # result.append(int(right_count == len(a) + len(b)))
        
        right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        a, b = merge_by_two_annotations(annotations1, annotations2, img1, img2)
        result.append(int(right_count == len(a) + len(b)))


        if right_count != len(a) + len(b):
            wrong.append(annotations1[0]['pair_id'])
        print 'right_count: ', right_count, 'predict_count: ', len(a) + len(b), 'acc: ', 1.0 * sum(result) / len(result), 'wrong: ', wrong

def test10():
    data = get_anno_data7()
    pair_annotations = []
    for i in range(1000):
        pair = [0,0]
        for img_id, d in data.items():
            if d[0]['pair_id'] == i:
                pair[d[0]['location'] % 2] = d
        if pair[0] and pair[1]:
            pair_annotations.append(pair)

    conn = MySQLdb.connect(host='192.168.1.12', user='root', passwd='reload123', db='shampoo')
    cur = conn.cursor()
    cur.execute('select object_id, same_object_id from object_sames_citybox_test9 where status = 1')
    data = cur.fetchall()
    object_id_pair = [int(i[0]) for i in data] + [int(i[1]) for i in data]
    
    cur.execute('select distinct(a.img_id) from object_sames_citybox_test9 os inner join object a on os.object_id =a.id inner join object b on os.same_object_id =b.id where a.type<>b.type and os.status = 1;')
    data = cur.fetchall()
    bug_img_ids = [int(i[0]) for i in data]
    
    result = []
    wrong = []
    for annotations1, annotations2 in pair_annotations:
        bad_case = [82, 126, 262, 367]
        #if annotations1[0]['pair_id'] not in bad_case:
        #    continue
        print annotations1[0]['pair_id']
        
        # right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        # img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        # a, b = merge_by_two_annotations(annotations1, annotations2, img1, img2, False)
        # result.append(int(right_count == len(a) + len(b)))
        
        right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        a, b = merge_by_two_annotations(annotations1, annotations2, img1, img2)
        result.append(int(right_count == len(a) + len(b)))


        if right_count != len(a) + len(b):
            wrong.append(annotations1[0]['pair_id'])
        print 'right_count: ', right_count, 'predict_count: ', len(a) + len(b), 'acc: ', 1.0 * sum(result) / len(result), 'wrong: ', wrong



init_siamese_net()
if __name__ == '__main__':
    test4()


 
