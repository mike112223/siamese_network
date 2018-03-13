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
    


       

    
    distinct_annotations1, distinct_annotations2, duplicate_pair_annos = merge_by_two_annotations(annotations1, annotations2, img1, img2, True)
    # distinct_annotations1, distinct_annotations2 = merge_by_two_annotations(annotations1, annotations2, img1, img2, False)
   
    for anno in annotations1:
        x0,y0,x1,y1 = anno['bbox']
        cv2.rectangle(img1, (x0,y0), (x1,y1), color_type[anno['type']], 3)
        cv2.putText(img1, "%s , %s" % (anno['center'][0], anno['center'][1]), (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_type[anno['type']], 4)

    for anno in annotations2:
        x0,y0,x1,y1 = anno['bbox']
        cv2.rectangle(img2, (x0,y0), (x1,y1), color_type[anno['type']], 3)
        cv2.putText(img2, "%s , %s" % (anno['center'][0], anno['center'][1]), (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_type[anno['type']], 4)

    img = np.zeros((h*2, w*2, 3))
    img[:h,:w] = img1 
    img[h:,w:] = img2 
    for anno1, anno2 in duplicate_pair_annos:
        cv2.line(img, (anno1['center'][0], anno1['center'][1]), (anno2['center'][0] + w, anno2['center'][1] + h), (np.random.rand(3) * 256).astype(int), 2)

    cv2.imwrite('./result4/%s.jpg' % annotations1[0]['pair_id'], img)

    test_img = np.zeros((h*2, w*2, 3))
    merge_img = np.zeros((h, w, 3))

    for anno in distinct_annotations1:
        x0,y0,x1,y1 = anno['bbox']
        test_img[y0:y1, x0:x1] = img1[y0:y1, x0:x1]
    
    for anno in distinct_annotations2:
        x0,y0,x1,y1 = anno['bbox']
        test_img[y0+h:y1+h, x0+w:x1+w] = img2[y0:y1, x0:x1]

    
    cv2.imwrite('./result4/%s_test.jpg' % annotations1[0]['pair_id'], test_img)

    return distinct_annotations1, distinct_annotations2


def merge_by_two_annotations(annotations1, annotations2, image1, image2, require_pairs = False):

    h, w = image1.shape[0:2]

    diff_value = 1.1
    type_penal_val = 2.0
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



    distinct_annotations1, inter_annotations1 = [], [i for i in annotations1]  # all crop objects are default in inter_set
    distinct_annotations2, inter_annotations2 = [], [i for i in annotations2] 
    
    anno_scaled_images = []
    for anno in inter_annotations1:
        #anno_scaled_bbox = scale_bbox(anno['bbox'], 2.5, min_len=120)
        anno_scaled_bbox = produce_bbox(anno['bbox'], 120, h, w)
        x1, y1, x2, y2 = anno_scaled_bbox
        anno_scaled_image = image1[y1: y2, x1: x2]
        anno_scaled_images.append(anno_scaled_image)

    anno_scaled_images_feature = get_siamese_feature(anno_scaled_images)  # feed into single input network to get feat vector
    for i in range(len(inter_annotations1)):  # tag every image with siamese_feature
        inter_annotations1[i]['siamese_feature'] = anno_scaled_images_feature[i]
    
    anno_scaled_images = []   # do the same thing to left one
    for anno in inter_annotations2:
        #anno_scaled_bbox = scale_bbox(anno['bbox'], 2.5, min_len=120)  # expand to original size X 2.5
        anno_scaled_bbox = produce_bbox(anno['bbox'], 120, h, w)
        x1, y1, x2, y2 = anno_scaled_bbox
        anno_scaled_image = image2[y1: y2, x1: x2]
        anno_scaled_images.append(anno_scaled_image)

    anno_scaled_images_feature = get_siamese_feature(anno_scaled_images)
    for i in range(len(inter_annotations2)):
        inter_annotations2[i]['siamese_feature'] = anno_scaled_images_feature[i]

    siamese_log = open("./siamese_log", 'w')
    len1, len2 = len(inter_annotations1), len(inter_annotations2)
    dist_map = np.zeros((len1, len2))
    for i in range(len1):
        for j in range(len2):
            anno1, anno2 = inter_annotations1[i], inter_annotations2[j]
            dist_map[i,j] = np.linalg.norm(anno1['siamese_feature'] - anno2['siamese_feature'])
            if anno1['type'] != anno2['type']:  # to make sure different type objects wont match, maybe with/height ratio can judge 
                dist_map[i,j] *= type_penal_val
            # if anno1['type'] != anno2['type'] and (anno1['score'] > 0.9 and anno2['score'] > 0.9):  # to make sure different type objects wont match
            #     dist_map[i,j] *= type_penal_val
            # if anno1['type'] == anno2['type'] and (anno1['score'] <= 0.9 or anno2['score'] <= 0.9):
            #     dist_map[i,j] *= type_penal_val
                # pass
            text = str(anno1['center'][0]) + "," + str(anno1['center'][1]) + "---" + str(anno2['center'][0]) + "," + str(anno2['center'][1]) + ":" \
            + str(dist_map[i, j]) + "\r\n"
            siamese_log.writelines(text)

    siamese_log.close()

    #left one
    dist_map2 = np.zeros((len2, len1))
    for j in range(len2):
        for i in range(len1):
            dist_map2[j, i] = dist_map[i, j]

    #print len1, "___", len2

    paired_index1 = []
    paired_index2 = []
    duplicate_pair_annos_index = []
    import heapq
    import math

    def exclude_pairs(i, j, width = 0):
        pair_anno1, pair_anno2 = inter_annotations1[i], inter_annotations2[j]
        for x in range(len1):
            for y in range(len2):
                if dist_map[x,y] > diff_value:
                    continue
                anno1, anno2 = inter_annotations1[x], inter_annotations2[y]  # if patch x in right area and patch y in left area, tag them different, and so on
                if ((anno1['bbox'][0] > pair_anno1['bbox'][2] + width) and (anno2['bbox'][2] < pair_anno2['bbox'][0] - width)) or \
                ((anno2['bbox'][0] > pair_anno2['bbox'][2] + width) and (anno1['bbox'][2] < pair_anno1['bbox'][0] - width)) or \
                ((anno1['bbox'][1] > pair_anno1['bbox'][3] + width) and (anno2['bbox'][3] < pair_anno2['bbox'][1] - width)) or \
                ((anno2['bbox'][1] > pair_anno2['bbox'][3] + width) and (anno1['bbox'][3] < pair_anno1['bbox'][1] - width)) :
                    if dist_map[x, y] == 50 or dist_map[x, y] == 100:
                        continue
                    dist_map[x, y] *= 2.0



    can_list_left = {}      # find left candidates
    for i in range(len1):
        candidates = []
        for j in range(len2):
            if dist_map[i, j] < diff_value:
                heapq.heappush(candidates, (-dist_map[i, j], j))
                if len(candidates) > 2:
                    heapq.heappop(candidates)
        can_list_left[i] = candidates

    can_list_right = {}     # find right candidates
    for j in range(len2):
        candidates = []
        for i in range(len1):
            if dist_map2[j, i] < diff_value:
                heapq.heappush(candidates, (-dist_map2[j, i], i))
                if len(candidates) > 2:
                    heapq.heappop(candidates)
        can_list_right[j] = candidates

    
    def print_pair(i, j):
        print str(inter_annotations1[i]['center'][0]) + "," + str(inter_annotations1[i]['center'][1]) + "________" + \
        str(inter_annotations2[j]['center'][0]) + "," + str(inter_annotations2[j]['center'][1]) + ":" + str(dist_map[i, j])

    to_confirm = {}
    while True:                          # find the easily recognized left&right images' common objects
        if dist_map.min() > diff_value:
            break
        min_index = np.where(dist_map==dist_map.min())  # find the most likely
        i, j = min_index[0][0], min_index[1][0]
        # print i, ",", j, "   :", dist_map[i, j]
        

        if len(can_list_left[i]) == 2 and can_list_left[i][1][0] - can_list_left[i][0][0] < 0.3:   # need to judge by range
            # print can_list_left[i][1][0], "____", can_list_left[i][0][0]
            # print dist_map[i, can_list_left[i][0][1]], "_____", dist_map[i, can_list_left[i][1][1]]
            # if j == can_list_left[i][1][1] or j == can_list_left[i][0][1]:
            if dist_map[i, can_list_left[i][0][1]] != 100 : dist_map[i, can_list_left[i][0][1]] = 50
            if dist_map[i, can_list_left[i][1][1]] != 100 : dist_map[i, can_list_left[i][1][1]] = 50
            if dist_map[i, j] != 100 : dist_map[i, j] = 50  # make sure wont be picked twice
            to_confirm[i] = [can_list_left[i][0][1], can_list_left[i][1][1]]
            continue
        # print str(inter_annotations1[i]['center'][0]) + "," + str(inter_annotations1[i]['center'][1]) + ":" + \
        # str(inter_annotations2[j]['center'][0]) + "," + str(inter_annotations2[j]['center'][1]) + ":" + str(dist_map[i, j])
        dist_map[i], dist_map[:,j] = 100, 100           # mark identified patch with other patches exclusive
        dist_map2[j], dist_map2[:, i] = 100, 100
        pair_anno1, pair_anno2 = inter_annotations1[i], inter_annotations2[j]
        # print str(inter_annotations1[i]['center'][0]) + "," + str(inter_annotations1[i]['center'][1])
        # print dist_map[i]
        # print str(inter_annotations2[j]['center'][0]) + "," + str(inter_annotations2[j]['center'][1])
        # print dist_map[:,j]


        exclude_pairs(i, j)
        duplicate_pair_annos_index.append([i, j])
        paired_index1.append(i)
        paired_index2.append(j)
        #print_pair(i, j)
        
    # print "***************************Hard pair*****************************"
    hard_confirm = {}

    key_index = list(to_confirm.keys())  # reverse to better recognize
    while True:                                     # update dist_map until it doesn't change
        change_flag = False
        for i in key_index:
            can_1 = to_confirm[i][0]
            can_2 = to_confirm[i][1]
            if dist_map[i, can_1] == 100 and dist_map[i, can_2] == 100:
                continue
            elif dist_map[i, can_1] == 50 and dist_map[i, can_2] == 100:   # need to check object relative pos
                #print "filtered right"
                #print_pair(i, can_1)
                #print_pair(i, can_2)
                exclude_pairs(i, can_1)
                duplicate_pair_annos_index.append([i, can_1])
                paired_index1.append(i)
                paired_index2.append(can_1)
                dist_map[i], dist_map[:,can_1] = 100, 100
                change_flag = True
            elif dist_map[i, can_2] == 50 and dist_map[i, can_1] == 100:
                #print "filtered left"
                #print_pair(i, can_1)
                #print_pair(i, can_2)
                exclude_pairs(i, can_2)
                duplicate_pair_annos_index.append([i, can_2])
                paired_index1.append(i)
                paired_index2.append(can_2)
                dist_map[i], dist_map[:,can_2] = 100, 100
                change_flag = True
            else:
                pass
        if not change_flag:
            break

    for i in key_index:
        can_1 = to_confirm[i][0]
        can_2 = to_confirm[i][1]
        if dist_map[i, can_1] == 50 and dist_map[i, can_2] == 50:
            #print "add hard pair"
            #print_pair(i, can_1)
            #print_pair(i, can_2)
            hard_confirm[i] = [can_1, can_2]

    

    key_index2 = list(hard_confirm.keys())
    # to_del = []
    # mix_pair = []
    # for i in key_index2:    # converge very same pair
    #     if i in to_del:
    #         continue
    #     for j in key_index2:
    #         if j in to_del or j == i:
    #             continue
    #         if i != j and (hard_confirm[i] == hard_confirm[j] or hard_confirm[i] == hard_confirm[j][::-1]):  
    #             to_del.append(i)
    #             to_del.append(j)
    #             mix_pair.append([(i, j), (hard_confirm[j])])  # so far all pair have been found




    # left_mix_pair = []
    # right_mix_pair = []
    # for left, right in mix_pair:
    #     left_mix_pair.append(left[0])
    #     left_mix_pair.append(left[1])
    #     right_mix_pair.append(right[0])
    #     right_mix_pair.append(right[1])
    #     duplicate_pair_annos_index.append([left[0], right[0]])
    #     duplicate_pair_annos_index.append([left[1], right[1]])


    # left_distinct = []
    # right_distinct = []
    # for i in [n for n in key_index2 if n not in to_del]:  # left distinct right distinct
    #     left_distinct.append(i)
    #     right_distinct.append(hard_confirm[i][0])
    #     right_distinct.append(hard_confirm[i][1])

    left_delete = []
    right_add = []
    # print "*_*_*_"
    # print hard_confirm," ", type(hard_confirm), " ", len(hard_confirm)
    for l_inx in hard_confirm:
        if l_inx not in left_delete:
            left_delete.append(l_inx)
        r_inx_0 = hard_confirm[l_inx][0]
        r_inx_1 = hard_confirm[l_inx][1]
        if r_inx_0 not in right_add:
            right_add.append(r_inx_0)
        if r_inx_1 not in right_add:
            right_add.append(r_inx_1)


    # print "*****************Mix pair******************************"
    # for i in left_delete:
    #     print str(inter_annotations1[i]['center'][0]) + "," + str(inter_annotations1[i]['center'][1])
    # print "_______"
    # for i in right_add:
    #     print str(inter_annotations2[i]['center'][0]) + "," + str(inter_annotations2[i]['center'][1])




    for i in range(len1):  # generate distinct objects
        if i not in paired_index1 and i not in left_delete:
            distinct_annotations1.append(inter_annotations1[i])

    for i in range(len2):
        if i not in paired_index2 or i in right_add:
            distinct_annotations2.append(inter_annotations2[i])
    
    for anno in inter_annotations1 + inter_annotations2:
        anno.pop('siamese_feature')


    duplicate_pair_annos = [[inter_annotations1[i[0]], inter_annotations2[i[1]]] for i in duplicate_pair_annos_index]
    for anno1, anno2 in duplicate_pair_annos:  # for common objects if types are different choose the score higher one 
        if anno1['type'] != anno2['type']:
            anno1['type'], anno2['type'] = [anno2['type'], anno2['type']] if anno2['score'] > anno1['score'] else [anno1['type'], anno1['type']]
        # if anno1['center'][0] - w * 0.26 < 0 and anno1 not in distinct_annotations1:
        if  anno1 not in distinct_annotations1:  # classify left area object into left group and do same thing to right area
            distinct_annotations1.append(anno1)
        # elif anno2 not in distinct_annotations2:
        #     distinct_annotations2.append(anno2)
    #print len(distinct_annotations1), "___", len(distinct_annotations2)
    #for i in distinct_annotations2:
        #print i['center'][0],",",i['center'][1]
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
        caffe.set_device(0)

        model_def = '/core1/data/home/zengcheng/siamese_shelf/res50/deploy.prototxt'  # same network just input one image output one vector
        # model_weights = '/core1/data/home/zengcheng/siamese_shelf/res50/resnet50_v6_iter_160000.caffemodel'
        model_weights = '/home/zhuyanjia/py-faster-rcnn/siamese_shelf/res50/models/resnet50_v1_iter_240000.caffemodel'
        # model_weights = '/home/zhoujin/siamese_shelf/res50/models/resnet50_v0_iter_200000.caffemodel'
       # model_weights = '/home/zhoujin/siamese_shelf/res50/models/resnet50_9_10_iter_160000.caffemodel'
       # model_weights = '/home/zhoujin/siamese_shelf/res50/models/resnet50_9_10_iter_170000.caffemodel'
        # model_weights = '/home/zhoujin/siamese_shelf/res50/models/resnet50_9_10_iter_180000.caffemodel'
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

def produce_bbox(bbox, length, h, w):

    scaled_bbox = [0,0,0,0]
    scaled_bbox[0] = max(bbox[0]-length, 0)
    scaled_bbox[1] = max(bbox[1]-length, 0)
    scaled_bbox[2] = min(w, bbox[2]+length)
    scaled_bbox[3] = min(h, bbox[3]+length)

    return [int(i) for i in scaled_bbox]

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

    data = cur.fetchall()

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

    data = cur.fetchall()

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

    data = cur.fetchall()

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
        # if i not in [15, 18, 40, 114, 124, 125, 221, 222, 283, 396, 587, 818, 879, 967]:
        #     continue
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
    
    cur.execute('select distinct(a.img_id) from object_sames_9 os inner join object a on os.object_id =a.id inner join object b on os.same_object_id =b.id where a.type<>b.type;')
    data = cur.fetchall()
    bug_img_ids = [int(i[0]) for i in data]
    
    import time
    start = time.time()
    result = []
    wrong = []
    for annotations1, annotations2 in pair_annotations: #[10:30]
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
            print annotations1[0]['path'], annotations2[0]['path']
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
    
    cur.execute('select distinct(a.img_id) from object_sames_10 os inner join object a on os.object_id =a.id inner join object b on os.same_object_id =b.id where a.type<>b.type;')
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
        a, b = merge_by_two_annotations(annotations1, annotations2, img1, img2)
        result.append(int(right_count == len(a) + len(b)))
        
        # right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        # img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        # a, b = merge_by_two_annotations_test(annotations1, annotations2, img1, img2)
        # result.append(int(right_count == len(a) + len(b)))


        if right_count != len(a) + len(b):
            wrong.append(annotations1[0]['pair_id'])
            print annotations1[0]['path'], annotations2[0]['path']
        print 'right_count: ', right_count, 'predict_count: ', len(a) + len(b), 'acc: ', 1.0 * sum(result) / len(result), 'wrong: ', wrong

def test4():
    data = get_anno_data2() # right data
    pair_annotations = []   
    for i in range(1000):
        # if i not in [63, 76, 82, 121, 130, 208, 209, 227, 274, 301, 334, 342, 352, 355, 367, 368, 379, 411, 416, 417, 428, 431, 444, 457, 499]:
        #     continue
        # if i not in [76, 121, 130, 141, 144, 148, 223, 227, 275, 334, 342, 345, 354, 367, 368, 404, 416, 428, 431, 444, 457]:
        #     continue
        # if i not in [63, 76, 82, 130, 144, 148, 219, 223, 227, 275, 342, 367, 368, 428, 457, 460, 465, 482, 499]:
        #     continue
        # if i not in [3, 11, 34, 64, 76, 86, 89, 144, 192, 193, 222, 227, 275, 289, 314, 342, 362, 368, 392, 410, 412, 457, 459, 484, 488]:
        #     continue
        pair = [0,0]
        for img_id, d in data.items():
            if d[0]['pair_id'] == i:
                pair[d[0]['location'] % 2] = d
        if pair[0] and pair[1]:
            pair_annotations.append(pair)  # test image patches

    conn = MySQLdb.connect(host='192.168.1.12', user='root', passwd='reload123', db='shampoo')
    cur = conn.cursor()
    cur.execute('select object_id, same_object_id from object_sames_11 where status = 1')
    data = cur.fetchall()
    object_id_pair = [int(i[0]) for i in data] + [int(i[1]) for i in data]
    
    # select hard case
    cur.execute('select distinct(a.img_id) from object_sames_11 os inner join object a on os.object_id =a.id inner join object b on os.same_object_id =b.id where a.type<>b.type;')
    data = cur.fetchall()
    bug_img_ids = [int(i[0]) for i in data]
    
    result = []
    wrong = []
    for annotations1, annotations2 in pair_annotations:
        bad_case = [34, 59, 90, 94, 109, 130, 140, 154, 219, 222, 239, 265, 338, 341, 374, 375, 404, 428, 460, 479, 482]
        if annotations1[0]['pair_id'] not in bad_case:
            continue
        print annotations1[0]['pair_id']
        
        # right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        # img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        # a, b = merge_by_two_annotations(annotations1, annotations2, img1, img2, False)
        # result.append(int(right_count == len(a) + len(b)))
        
        right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        a, b = merge_by_two_annotations_test(annotations1, annotations2, img1, img2)
        result.append(int(right_count == len(a) + len(b)))


        if right_count != len(a) + len(b):
            wrong.append(annotations1[0]['pair_id'])
            print annotations1[0]['path'], annotations2[0]['path']
        print 'right_count: ', right_count, 'predict_count: ', len(a) + len(b), 'acc: ', 1.0 * sum(result) / len(result), 'wrong: ', wrong

def test5():
    import json
    with open('./debug.json') as f:
        debug = json.loads(f.read())
    with open('./debug_output_v5.json') as f:
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
        print annotations1[0]['pair_id']

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
        print annotations1[0]['pair_id']

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
        print annotations1[0]['pair_id']

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
        print annotations1[0]['pair_id']

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
    
