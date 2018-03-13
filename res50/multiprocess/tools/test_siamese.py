import os
import sys
sys.path.insert(0, '/core1/data/deploy/caffe.git/python')
import cv2
import caffe
# import shutil
import numpy as np
import json
from sklearn.preprocessing import normalize



def _get_transformer(shape, mean, lib='cf'):
    transformer = caffe.io.Transformer({'data':shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', mean)

    if lib == 'cf':
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2,1,0))

    return transformer

def softmax(x):

    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def crop_image(item, length):
	path = item['img_path']
	bbox = [int(item['xmin']), int(item['ymin']), int(item['xmin'])+int(item['width']), int(item['ymin'])+int(item['height'])]
	img = cv2.imread(path)
	h, w = img.shape[0:2]
	bbox[0] = max(0, bbox[0]-length)
	bbox[1] = max(0, bbox[1]-length)
	bbox[2] = min(w, bbox[2]+length)
	bbox[3] = min(h, bbox[3]+length)
	return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
'''
def crop_image(item, scale_size):
	min_len = 120
	path = item['img_path']
	bbox = [int(item['xmin']), int(item['ymin']), int(item['xmin'])+int(item['width']), int(item['ymin'])+int(item['height'])]
	img = cv2.imread(path)

	scaled_bbox = [0,0,0,0]
	widths = bbox[2] - bbox[0] + 1.0
	heights = bbox[3] - bbox[1] + 1.0
	ctr_x = bbox[0] + 0.5 * widths
	ctr_y = bbox[1] + 0.5 * heights

	if min_len:
	    scale_size = max(scale_size, min_len/widths, min_len/heights)

	scaled_widths = widths * scale_size
	scaled_heights = heights * scale_size

	scaled_bbox[0] = int(max(0, ctr_x - 0.5 * scaled_widths))
	scaled_bbox[2] = int(ctr_x + 0.5 * scaled_widths)
	scaled_bbox[1] = int(max(0, ctr_y - 0.5 * scaled_heights))
	scaled_bbox[3] = int(ctr_y + 0.5 * scaled_heights)

	return img[scaled_bbox[1]:scaled_bbox[3], scaled_bbox[0]:scaled_bbox[2]]
'''

caffe.set_mode_gpu()
caffe.set_device(0) 

model_def = '../../merge_models/ResNet_50_deploy.prototxt'
model_weights = '../../models/resnet50_v2_iter_220000.caffemodel'
# model_weights = '/core1/data/home/shizhan/jiahuan/siamese_shelf/res50/models/resnet50_magic_loss_iter_9600.caffemodel'

net = caffe.Net(model_def, model_weights, caffe.TEST)
net.blobs['data'].reshape(1, 3, 224, 224)
transformer = _get_transformer(net.blobs['data'].data.shape, np.array([104, 117, 123]), 'cv')

###
objectid_to_metadata = json.load(open('/data-4t/home/yanjia/siamese_shelf/length_data/json_data/objectid_to_metadata.json','r'))
with open('/home/zhuyanjia/py-faster-rcnn/siamese_shelf/res50/multiprocess/test/length_image_list_test_shelf_9_10_11.txt') as f:
    data1 = [i.strip().split() for i in f]
with open('/home/zhuyanjia/py-faster-rcnn/siamese_shelf/res50/multiprocess/test/length_image_list_test_shelf_9_10_11_p.txt') as f:
    data2 = [i.strip().split() for i in f]
###

blen = 0
ind = 0
for i in range(len(data1)):
	if data1[i][2] == '90':
		blen = i
		break

for i in range(len(data1)):
	if data1[i][2] == '120':
		ind = i 
		break


labels = []
distances = []
right_count = []
file_path = []
img_list = []
obj_id = []
pair_id = 0
for j in range(blen):
	i = j+ind
	object_id1, label, l = data1[i]
	object_id2 = data2[i][0]
	if label == 'none':
		if labels and 1 in labels:
			print '-------------------------', labels.index(1), distances.index(min(distances))
			right_count.append(int(labels.index(1)==distances.index(min(distances))))
			print 1.0 * sum(right_count) / len(right_count)
			if labels.index(1) != distances.index(min(distances)):
				#for i in [labels.index(1), distances.index(min(distances))]:
				#	print img_list[i][0].shape
				#	print img_list[i][1].shape
				#	cv2.imwrite('../bad_ex/%d-%s.jpg'%(pair_id, obj_id[i][0]), img_list[i][0])
				#	cv2.imwrite('../bad_ex/%d-%s-%.2f.jpg'%(pair_id, obj_id[i][1], distances[i]), img_list[i][1])
				##for path in file_path[labels.index(1)] + file_path[distances.index(min(distances))]:
					#shutil.copy(path, '../bad_examples/')
				pair_id += 1

		file_path = []
		labels = []
		distances = []
		img_list = []
		obj_id = []
		continue

	object1 = objectid_to_metadata[object_id1]
	object2 = objectid_to_metadata[object_id2]
	ipath1 = object1['img_path']
	ipath2 = object2['img_path']
	image1 = crop_image(object1, int(l))
	image2 = crop_image(object2, int(l))
#	img_list.append([image1, image2])
	obj_id.append([object_id1, object_id2])

	image1, image2 = [transformer.preprocess('data', image1)], [transformer.preprocess('data', image2)]
	net.blobs['data'].data[...] = image1
	net.blobs['data_p'].data[...] = image2
	net.forward()
	vec1, vec2 = net.blobs['feat'].data.copy(), net.blobs['feat_p'].data.copy()
	dist = np.linalg.norm(vec1 - vec2)
	distances.append(dist)
	print ipath1, object_id1, ipath2, object_id2, label, dist
	labels.append(int(label))
	file_path.append([ipath1, ipath2])
