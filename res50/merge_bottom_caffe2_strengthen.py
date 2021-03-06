import os
os.environ['GLOG_minloglevel'] = '2'
import cv2
import math
import numpy as np
import MySQLdb
import pickle 

from caffe2.python import workspace
import time




def merge_by_two_annotations_test(annotations1, annotations2, img1, img2, num):
    h, w = img1.shape[0:2]
    color_type = {i: (np.random.rand(3) * 256).astype(int) for i in range(10000)}
    


    
    distinct_annotations1, distinct_annotations2, duplicate_pair_annos = merge_by_two_annotations(annotations1, annotations2, img1, img2, True, net, device_opts, num)
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


def merge_by_two_annotations(annotations1, annotations2, image1, image2, require_pairs = False, net=None, device_opts=None, num=None):

    h, w = image1.shape[0:2]

    diff_value = 1.55
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

    anno_scaled_images_feature = get_siamese_feature(net, anno_scaled_images, device_opts=device_opts)  # feed into single input network to get feat vector
    for i in range(len(inter_annotations1)):  # tag every image with siamese_feature
        inter_annotations1[i]['siamese_feature'] = anno_scaled_images_feature[i]
    
    anno_scaled_images = []   # do the same thing to left one
    for anno in inter_annotations2:
        #anno_scaled_bbox = scale_bbox(anno['bbox'], 2.5, min_len=120)  # expand to original size X 2.5
        anno_scaled_bbox = produce_bbox(anno['bbox'], 120, h, w)
        x1, y1, x2, y2 = anno_scaled_bbox
        anno_scaled_image = image2[y1: y2, x1: x2]
        anno_scaled_images.append(anno_scaled_image)

    anno_scaled_images_feature = get_siamese_feature(net, anno_scaled_images, device_opts=device_opts)
    for i in range(len(inter_annotations2)):
        inter_annotations2[i]['siamese_feature'] = anno_scaled_images_feature[i]

    if num != None:
        siamese_log = open("./result4/siamese_log_%d.txt"%num,"w")
    else:
        siamese_log = open("./result4/siamese_log.txt", 'w')
    len1, len2 = len(inter_annotations1), len(inter_annotations2)
    dist_map = np.zeros((len1, len2))
    for i in range(len1):
        for j in range(len2):
            anno1, anno2 = inter_annotations1[i], inter_annotations2[j]
            dist_map[i,j] = np.linalg.norm(anno1['siamese_feature'] - anno2['siamese_feature'])
            if anno1['type'] != anno2['type'] and (anno1['score'] > 0.7 and anno2['score'] > 0.7):  # to make sure different type objects wont match
                dist_map[i,j] *= type_penal_val
            if anno1['type'] == anno2['type'] and ((anno1['score'] <= 0.7 and anno2['score'] > 0.7) or (anno1['score'] > 0.7 and anno2['score'] <= 0.7)):
                dist_map[i,j] *= type_penal_val
            # if anno1['type'] != anno2['type']:
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

    # print len1, "___", len2

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

    def update_dist_map(i, j):
        exclude_pairs(i, j)
        duplicate_pair_annos_index.append([i, j])
        paired_index1.append(i)
        paired_index2.append(j)
        dist_map[i], dist_map[:,j] = 100, 100


    can_list_left = {}      # find left candidates
    backup_list = {}
    for i in range(len1):
        candidates = []
        for j in range(len2):
            if dist_map[i, j] < diff_value:
                heapq.heappush(candidates, (-dist_map[i, j], j))
                if len(candidates) > 3:
                    heapq.heappop(candidates)
        if len(candidates) == 3:
            backup = candidates.pop(0)
            backup_list[i] = backup[1]
        can_list_left[i] = candidates
        

    # can_list_right = {}     # find right candidates
    # for j in range(len2):
    #     candidates = []
    #     for i in range(len1):
    #         if dist_map2[j, i] < diff_value:
    #             heapq.heappush(candidates, (-dist_map2[j, i], i))
    #             if len(candidates) > 2:
    #                 heapq.heappop(candidates)
    #     can_list_right[j] = candidates

    
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
        
        if (len(can_list_left[i]) == 2 and abs(can_list_left[i][1][0] - can_list_left[i][0][0]) < 0.3):   # need to judge by range
            # print can_list_left[i][1][1], "____", can_list_left[i][0][1]
            # print j
            # print dist_map[i, can_list_left[i][0][1]], "_____", dist_map[i, can_list_left[i][1][1]]
            if j == can_list_left[i][1][1] or j == can_list_left[i][0][1]:
                if dist_map[i, can_list_left[i][0][1]] != 100 : dist_map[i, can_list_left[i][0][1]] = 50
                if dist_map[i, can_list_left[i][1][1]] != 100 : dist_map[i, can_list_left[i][1][1]] = 50
                to_confirm[i] = [can_list_left[i][0][1], can_list_left[i][1][1]]
                # print "add to_confirm"
                # print_pair(i, can_list_left[i][0][1])
                # print_pair(i, can_list_left[i][1][1])
                # print "end"
                continue
            if i in to_confirm:
                if dist_map[i, j] != 100 : dist_map[i, j] = 50  # tag it unconfirmed
                continue
        # print str(inter_annotations1[i]['center'][0]) + "," + str(inter_annotations1[i]['center'][1]) + ":" + \
        # str(inter_annotations2[j]['center'][0]) + "," + str(inter_annotations2[j]['center'][1]) + ":" + str(dist_map[i, j])
        dist_map[i], dist_map[:,j] = 100, 100           # mark identified patch with other patches exclusive
        dist_map2[j], dist_map2[:, i] = 100, 100
        # print str(inter_annotations1[i]['center'][0]) + "," + str(inter_annotations1[i]['center'][1])
        # print dist_map[i]
        # print str(inter_annotations2[j]['center'][0]) + "," + str(inter_annotations2[j]['center'][1])
        # print dist_map[:,j]


        exclude_pairs(i, j)
        duplicate_pair_annos_index.append([i, j])
        paired_index1.append(i)
        paired_index2.append(j)
        # print_pair(i, j)
        
    # print "***************************Hard pair*****************************"
    hard_confirm = {}

    key_index = list(to_confirm.keys())  # reverse to better recognize
    while True:                                     # update dist_map until it doesn't change
        change_flag = False
        for i in key_index:
            if len(to_confirm[i]) == 0:
                continue
            elif len(to_confirm[i]) == 1:
                can = to_confirm[i][0]
                backup = backup_list.get(i, None)

                # print "backup: ", backup
                if backup == None:
                    if dist_map[i, can] != 100:
                        # print "hard pair : ", print_pair(i, can)
                        update_dist_map(i, can)
                    to_confirm[i].pop()
                elif backup != None and (dist_map2[backup, i] - dist_map2[can, i] > 0.2 or dist_map[i, backup] == 100):
                    if dist_map[i, can] != 100:
                        # print "hard pair 2 : ", print_pair(i, can)
                        update_dist_map(i, can)
                    to_confirm[i].pop()
                else:
                    # print "add backup:", print_pair(i, backup)
                    if dist_map[i, can] == 100: 
                        to_confirm[i].pop()
                    dist_map[i, backup] = 50
                    to_confirm[i].append(backup)
                    backup_list[i] = None
                change_flag = True
            elif len(to_confirm[i]) == 2:
                can_0 = to_confirm[i][0]
                can_1 = to_confirm[i][1]
                # print "handle double case:"
                # print_pair(i, can_0)
                # print_pair(i, can_1)
                # print "end"
                # print to_confirm[i], " ", can_0, " ", can_1
                if dist_map[i, can_1] == 100:
                    to_confirm[i].pop(to_confirm[i].index(can_1))
                    change_flag = True
                if dist_map[i, can_0] == 100:
                    to_confirm[i].pop(to_confirm[i].index(can_0))
                    change_flag = True
            else:
                pass
            # can_1 = to_confirm[i][0]
            # can_2 = to_confirm[i][1]
            # if dist_map[i, can_1] == 100 and dist_map[i, can_2] == 100:
            #     continue
            # elif dist_map[i, can_1] == 50 and dist_map[i, can_2] == 100:   # need to check object relative pos
            #     # print "filtered right"
            #     # print_pair(i, can_1)
            #     # print_pair(i, can_2)
            #     exclude_pairs(i, can_1)
            #     duplicate_pair_annos_index.append([i, can_1])
            #     paired_index1.append(i)
            #     paired_index2.append(can_1)
            #     dist_map[i], dist_map[:,can_1] = 100, 100
            #     change_flag = True
            # elif dist_map[i, can_2] == 50 and dist_map[i, can_1] == 100:
            #     # print "filtered left"
            #     # print_pair(i, can_1)
            #     # print_pair(i, can_2)
            #     exclude_pairs(i, can_2)
            #     duplicate_pair_annos_index.append([i, can_2])
            #     paired_index1.append(i)
            #     paired_index2.append(can_2)
            #     dist_map[i], dist_map[:,can_2] = 100, 100
            #     change_flag = True
            # else:
            #     pass
        if not change_flag:
            break

    # calculate right and left mix_pairs
    left_mix_object = [[] for i in range(len(to_confirm))]
    right_mix_pair = []
    for i in key_index:
        if len(to_confirm[i]) < 2:
            continue
        can_1 = to_confirm[i][0]
        can_2 = to_confirm[i][1]
        if dist_map[i, can_1] == 50 and dist_map[i, can_2] == 50:
            # print "add hard pair"
            # print_pair(i, can_1)
            # print_pair(i, can_2)
            hard_confirm[i] = sorted([can_1, can_2])
            if hard_confirm[i] not in right_mix_pair:
                right_mix_pair.append(hard_confirm[i])
                left_mix_object[len(right_mix_pair)-1] = [i]
            else:
                left_mix_object[right_mix_pair.index(hard_confirm[i])].append(i)


    left_add = []
    right_delete = []
    for inx, l_mix in enumerate(left_mix_object):  # merge into left
        if len(l_mix) > 2:
            for i in l_mix:
                if i not in left_add:
                    left_add.append(i)
            for i in right_mix_pair[inx]:
                if i not in right_delete:
                    right_delete.append(i)



    key_index2 = list(hard_confirm.keys())
  

    left_delete = []
    right_add = []
    # print "*_*_*_"
    # print hard_confirm," ", type(hard_confirm), " ", len(hard_confirm)
    for l_inx in hard_confirm:
        if l_inx not in left_delete and l_inx not in left_add:
            left_delete.append(l_inx)
        r_inx_0 = hard_confirm[l_inx][0]
        r_inx_1 = hard_confirm[l_inx][1]
        if r_inx_0 not in right_add and r_inx_0 not in right_delete:
            right_add.append(r_inx_0)
        if r_inx_1 not in right_add and r_inx_1 not in right_delete:
            right_add.append(r_inx_1)


    # print "*****************Mix pair******************************"
    # for i in left_delete:
    #     print str(inter_annotations1[i]['center'][0]) + "," + str(inter_annotations1[i]['center'][1])
    # print "_______"
    # for i in right_add:
    #     print str(inter_annotations2[i]['center'][0]) + "," + str(inter_annotations2[i]['center'][1])




    for i in range(len1):  # generate distinct objects
        if (i not in paired_index1 and i not in left_delete) or i in left_add:
            distinct_annotations1.append(inter_annotations1[i])

    for i in range(len2):
        if (i not in paired_index2 and i not in right_delete) or i in right_add:
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
    # print len(distinct_annotations1), "___", len(distinct_annotations2)
    # for i in distinct_annotations2:
    #     print i['center'][0],",",i['center'][1]
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
device_opts = None

def init_siamese_net():
    global net, device_opts
    from caffe2.proto import caffe2_pb2
    from caffe2.python import core, workspace
    # sia_init_net = script_root + '/data/v7/c2_siamese/init_net.pb'
    # sia_predict_net = script_root + '/data/v7/c2_siamese/predict_net.pb'
    sia_init_net = '/core1/data/home/liuhuawei/tools/caffe2/build/init_net_v1_24W.pb'
    sia_predict_net = '/core1/data/home/liuhuawei/tools/caffe2/build/predict_net_v1_24W.pb'

    device_opts = core.DeviceOption(caffe2_pb2.CUDA, 1)
    init_def = caffe2_pb2.NetDef()
    with open(sia_init_net, 'r') as f:
        init_def.ParseFromString(f.read())
        init_def.device_option.CopyFrom(device_opts)
        workspace.RunNetOnce(init_def)

    net_def = caffe2_pb2.NetDef()
    with open(sia_predict_net, 'r') as f:
        net_def.ParseFromString(f.read())
        net_def.device_option.CopyFrom(device_opts)
        workspace.CreateNet(net_def, overwrite=True)
    net = net_def
    print("Finish loading siamese model!")


def cv_transform_img(img, resize_h=224, resize_w=224, mean=np.array([104.0, 117.0, 123.0], dtype=np.float32).reshape((1, 1, -1))):
    img = cv2.resize(img, (resize_w, resize_h))
    if mean is not None:
        img -= mean
    img = img.transpose((2, 0, 1))
    return img


def get_siamese_feature(net, images, batch_size=32, device_opts=None):
    features = []
    st = time.time()
    while len(images):
        if len(images) > batch_size:
            imgs = [cv_transform_img(images.pop(0).astype(np.float32)) for i in range(batch_size)]
            # print "transfomer", time.time() - st
            # print len(images)
            st = time.time()
            workspace.FeedBlob('data', np.array(imgs, dtype=np.float32), device_opts)
            workspace.RunNet(net.name, 1)
            output = workspace.FetchBlob('feat')
            # print "forward", time.time() - st
            st = time.time()
            features.extend([output[i].squeeze() for i in range(output.shape[0])])
        else:
            imgs = [cv_transform_img(images.pop(0).astype(np.float32)) for i in range(len(images))]
            # print "transfomer", time.time() - st
            st = time.time()
            workspace.FeedBlob('data', np.array(imgs, dtype=np.float32), device_opts)
            workspace.RunNet(net.name, 1)
            output = workspace.FetchBlob('feat')
            # print output.shape
            # print "forward", time.time() - st
            st = time.time()
            features.extend([output[i].squeeze() for i in range(output.shape[0])])
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

def test5():
    import json
    with open('/core1/data/home/samson/box_show/sku_detect_c2/data/debug_23.json') as f:
        debug = json.loads(f.read())
    with open('/core1/data/home/samson/box_show/sku_detect_c2/data/debug_23.json.anno.2') as f:
        json_lines = f.readlines()
    l = len(debug)
    lanno = len(json_lines)
    debug_output = json.loads(json_lines[0])
    l_a = len(debug_output)
    print l, "    ", lanno, "  ", l_a
    for i in [0]:
        debug_output = json.loads(json_lines[i])
        for cam_id in ['a']:
            print cam_id, i
            inx = None
            if cam_id == 'a':
                inx = 0
            else:
                inx = 2
            img1, img2 = cv2.imread(debug[i]['%s_cam_0' % cam_id]), cv2.imread(debug[i]['%s_cam_1' % cam_id])     
            annotations1, annotations2 = debug_output[inx], debug_output[inx+1]
            for a in annotations1 + annotations2:
                a['pair_id'] = cam_id + str(i)
            a, b = merge_by_two_annotations_test(annotations1, annotations2, img1, img2)

def get_anno_data_falldown():
    conn = MySQLdb.connect(host='192.168.1.12', user='root', passwd='reload123', db='shampoo')
    cur = conn.cursor()
    cur.execute('''
        select o.x_pixel, o.y_pixel, o.width_pixel, o.height_pixel, o.img_id, o.type, i.url, o.id from object o inner join image i on o.img_id = i.id where i.src_src in ('fall_down_test1')  and o.box_status=0 and i.status in (4,7);''')
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


def test4():
    data = get_anno_data2() # right data
    pair_annotations = []   
    for i in range(278,279):
        #if i not in [264, 367, 428]:
        #    continue
        pair = [0,0]
        for img_id, d in data.items():
            if d[0]['pair_id'] == i:
                pair[d[0]['location'] % 2] = d
        if pair[0] and pair[1]:
            pair_annotations.append(pair)  # test image patches
    
    #print pair_annotations
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
        #bad_case = [82, 130, 219, 264, 348, 367, 476]
        #if annotations1[0]['pair_id'] not in bad_case:
        #    continue
        print annotations1[0]['pair_id']
        
        # right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        # img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        # a, b = merige_by_two_annotations(annotations1, annotations2, img1, img2, False)
        # result.append(int(right_count == len(a) + len(b)))
        
        right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        #a, b = merge_by_two_annotations(annotations1, annotations2, img1, img2, False, net, device_opts)
        a, b = merge_by_two_annotations_test(annotations1, annotations2, img1, img2, annotations1[0]['pair_id'])
        result.append(int(right_count == len(a) + len(b)))


        if right_count != len(a) + len(b):
            wrong.append(annotations1[0]['pair_id'])
            print annotations1[0]['path'], annotations2[0]['path']
        print 'right_count: ', right_count, 'predict_count: ', len(a) + len(b), 'acc: ', 1.0 * sum(result) / len(result), 'wrong: ', wrong

def test1():
    data = get_anno_data()
    pair_annotations = []
    for i in range(1000):
        # if i not in [18, 40, 114, 124, 125, 221, 222, 283, 396, 587, 818, 879, 967]:
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
        a, b = merge_by_two_annotations(annotations1, annotations2, img1, img2, False, net, device_opts)
        #a, b = merge_by_two_annotations_test(annotations1, annotations2, img1, img2)
        result.append(int(right_count == len(a) + len(b)))


        if right_count != len(a) + len(b):
            wrong.append(annotations1[0]['pair_id'])
            print annotations1[0]['path'], annotations2[0]['path']
        print 'right_count: ', right_count, 'predict_count: ', len(a) + len(b), 'acc: ', 1.0 * sum(result) / len(result), 'wrong: ', wrong


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
        #a, b = merge_by_two_annotations_test(annotations1, annotations2, img1, img2)
        a, b = merge_by_two_annotations(annotations1, annotations2, img1, img2, False, net, device_opts)
        result.append(int(right_count == len(a) + len(b)))
        
        # right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        # img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        # a, b = merge_by_two_annotations_test(annotations1, annotations2, img1, img2)
        # result.append(int(right_count == len(a) + len(b)))


        if right_count != len(a) + len(b):
            wrong.append(annotations1[0]['pair_id'])
            print annotations1[0]['path'], annotations2[0]['path']
        print 'right_count: ', right_count, 'predict_count: ', len(a) + len(b), 'acc: ', 1.0 * sum(result) / len(result), 'wrong: ', wrong


def testfall():
    data = get_anno_data_falldown()
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
    # cur.execute('select object_id, same_object_id from object_sames_10 where status = 1')
    # data = cur.fetchall()
    # object_id_pair = [int(i[0]) for i in data] + [int(i[1]) for i in data]
    
    cur.execute('select distinct(img_id) from object o inner join image i on o.img_id=i.id where i.src_src = \'fall_down_test1\' and o.box_status=0 and i.status in (4,7);')
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
        
        # right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        a, b = merge_by_two_annotations_test(annotations1, annotations2, img1, img2)
        # result.append(int(right_count == len(a) + len(b)))
        
        # right_count = len(annotations1) + len(annotations2) - len([o for o in annotations1 if o['object_id'] in object_id_pair])
        # img1, img2 = cv2.imread(annotations1[0]['path']), cv2.imread(annotations2[0]['path'])
        # a, b = merge_by_two_annotations_test(annotations1, annotations2, img1, img2)
        # result.append(int(right_count == len(a) + len(b)))


        # if right_count != len(a) + len(b):
        #     wrong.append(annotations1[0]['pair_id'])
        #     print annotations1[0]['path'], annotations2[0]['path']
        print 'predict_count: ', len(a) + len(b)

init_siamese_net()
if __name__ == '__main__':
    test4() 
    
