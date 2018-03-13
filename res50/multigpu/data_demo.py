import json
import cv2
import random


data = json.load(open('/data-4t/home/yanjia/siamese_shelf/length_data/json_data/test.json','r'))



#for i in data.keys():
#	img_path_a = data[i]['path_a']
#	img_path_b = data[i]['path_b']

#	print img_path_a
#	print img_path_b

for i in range(1):
	key = data.keys()[random.randint(0,1000)]
	img_path_a = data[key]['path_a']
	print img_path_a
	img_path_b = data[key]['path_b']
	print img_path_b
	img_a = cv2.imread(img_path_a)
	img_b = cv2.imread(img_path_b)
	cv2.imwrite('./demo/000.jpg', img_a)
	cv2.imwrite('./demo/111.jpg', img_b)

	length = len(data[key]['bbox_a'])
	for i in range(length):
		bbox = data[key]['bbox_a'][i]
		bb = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
		img = img_a[bb[1]:bb[3], bb[0]:bb[2]]
		cv2.imwrite('./demo/0-%d.jpg'%(i), img)
	for i in range(length):
	 	bbox = data[key]['bbox_b'][i]
		bb = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
		img = img_b[bb[1]:bb[3], bb[0]:bb[2]]
		cv2.imwrite('./demo/1-%d.jpg'%(i), img)


