# coding=utf-8
import json
import MySQLdb
import numpy as np
import random


conn = MySQLdb.connect(host='192.168.1.12', user='root', passwd='reload123', db='shampoo')
cur = conn.cursor()

cur.execute('select object_id, same_object_id from object_sames where status = 1')
same_objects_id = [[int(i[0]), int(i[1])] for i in cur.fetchall()]

cur.execute('select object_id, same_object_id from object_sames_9 where status = 1')
same_objects_id += [[int(i[0]), int(i[1])] for i in cur.fetchall()]

cur.execute('select object_id, same_object_id from object_sames_10 where status = 1')
same_objects_id += [[int(i[0]), int(i[1])] for i in cur.fetchall()]

cur.execute('select object_id, same_object_id from object_sames_11 where status = 1')
same_objects_id += [[int(i[0]), int(i[1])] for i in cur.fetchall()]

cur.execute('select object_id, same_object_id from object_sames_citybox_test3 where status = 1')
same_objects_id += [[int(i[0]), int(i[1])] for i in cur.fetchall()]

cur.execute('select object_id, same_object_id from object_sames_citybox_test6 where status = 1')
same_objects_id += [[int(i[0]), int(i[1])] for i in cur.fetchall()]

cur.execute('select object_id, same_object_id from object_sames_citybox_test7 where status = 1')
same_objects_id += [[int(i[0]), int(i[1])] for i in cur.fetchall()]

cur.execute('select object_id, same_object_id from object_sames_citybox_test8 where status = 1')
same_objects_id += [[int(i[0]), int(i[1])] for i in cur.fetchall()]

cur.execute('select object_id, same_object_id from object_sames_citybox_test8_2 where status = 1')
same_objects_id += [[int(i[0]), int(i[1])] for i in cur.fetchall()]

cur.execute('select object_id, same_object_id from object_sames_citybox_test9 where status = 1')
same_objects_id += [[int(i[0]), int(i[1])] for i in cur.fetchall()]

cur.execute('select object_id, same_object_id from object_sames_citybox_test10 where status = 1')
same_objects_id += [[int(i[0]), int(i[1])] for i in cur.fetchall()]

same_objects_id = [[str(i[0]), str(i[1])] for i in same_objects_id]
same_objects_id = {'-'.join(i):1 for i in same_objects_id}
cur.execute('''select o.id, o.x_pixel, o.y_pixel, o.width_pixel, o.height_pixel, i.url, i.id, o.type, i.src_src from object o inner join image i on o.img_id = i.id where i.verify_status = 1 and o.box_status = 0 \
    and i.id in (select distinct img_id from object where id in (select object_id from object_sames where status = 1) \
    or o.id in (select object_id from object_sames_9 where status = 1 ) \
    or o.id in (select object_id from object_sames_10 where status = 1) \
    or o.id in (select object_id from object_sames_11 where status = 1) \
    or o.id in (select object_id from object_sames_citybox_test3 where status = 1) \
    or o.id in (select object_id from object_sames_citybox_test6 where status = 1) \
    or o.id in (select object_id from object_sames_citybox_test7 where status = 1) \
    or o.id in (select object_id from object_sames_citybox_test8 where status = 1) \
    or o.id in (select object_id from object_sames_citybox_test8_2 where status = 1) \
    or o.id in (select object_id from object_sames_citybox_test9 where status = 1) \
    or o.id in (select object_id from object_sames_citybox_test10 where status = 1) \
    or o.id in (select same_object_id from object_sames where status = 1) \
    or o.id in (select same_object_id from object_sames_9 where status = 1) \
    or o.id in (select same_object_id from object_sames_10 where status = 1) \
    or o.id in (select same_object_id from object_sames_11 where status = 1) \
    or o.id in (select same_object_id from object_sames_citybox_test3 where status = 1) \
    or o.id in (select same_object_id from object_sames_citybox_test6 where status = 1) \
    or o.id in (select same_object_id from object_sames_citybox_test7 where status = 1) \
    or o.id in (select same_object_id from object_sames_citybox_test8 where status = 1) \
    or o.id in (select same_object_id from object_sames_citybox_test8_2 where status = 1) \
    or o.id in (select same_object_id from object_sames_citybox_test9 where status = 1) \
    or o.id in (select same_object_id from object_sames_citybox_test10 where status = 1))''')

raw_data = [[int(i[0]), int(i[1]), int(i[2]), int(i[3]), int(i[4]), i[5], int(i[6]), int(i[7]), i[8]] for i in cur.fetchall()]
objects = {int(o[0]): {'bbox': [int(o[1]), int(o[2]), int(o[1] + o[3]), int(o[2] + o[4])], 'object_id': o[0], 'image_id': o[6], 'path': o[5], 'type': o[7]} for o in raw_data}

image_objects = {}
for d in raw_data:
    if d[6] in image_objects:
        image_objects[d[6]].append(d[0])
    else:
        image_objects[d[6]] = [d[0]]

all_images_pathes = {d[6]: d[5] for d in raw_data}

pair_images_id = {}

for d in raw_data:
    pair_id, location = d[5].split('_')[-2] + d[8].split('_')[-1], int(d[5].split('_')[-1].split('.')[0]) % 2
    if pair_id in pair_images_id:
        pair_images_id[pair_id][location] = d[6]
    else:
        pair_images_id[pair_id] = [0,0]
        pair_images_id[pair_id][location] = d[6]

###
print len(pair_images_id.values())


count = 0
max_diff_sku_pairs = 8
pair_images_id_list = pair_images_id.values()
random.shuffle(pair_images_id_list)
train_pair_images_id_list = pair_images_id_list[0:int(len(pair_images_id_list)*0.95)]
test_pair_images_id_list = pair_images_id_list[int(len(pair_images_id_list)*0.95):]

pair_train_data = {}
for idx, (image_id1, image_id2) in enumerate(train_pair_images_id_list):
    if image_id1 not in image_objects or image_id2 not in image_objects:
        continue
    image_objects_ids1 = image_objects.get(image_id1, [])
    image_objects_ids2 = image_objects.get(image_id2, [])
    if image_id1 not in all_images_pathes or image_id2 not in all_images_pathes:
        continue

    random.shuffle(image_objects_ids1)
    for object_id1 in image_objects_ids1:
        same_object_ids2 = [object_id2 for object_id2 in image_objects_ids2 if int(same_objects_id.get('-'.join([str(object_id1), str(object_id2)]), 0) or same_objects_id.get('-'.join([str(object_id2), str(object_id1)]), 0) )]
        if not same_object_ids2:
            continue

        image_objects_ids2.sort(key=lambda object_id2: np.linalg.norm(np.array(objects[object_id2]['bbox']) - np.array(objects[same_object_ids2[0]]['bbox'])))
        near_image_objects_ids2 = image_objects_ids2[0:max_diff_sku_pairs+1]
        for object_id2 in image_objects_ids2:
            object1 = objects[object_id1]
            object2 = objects[object_id2]

            label = int(object_id2 in same_object_ids2)

            if not label and object1['type'] != object2['type'] and object_id2 not in near_image_objects_ids2:
                continue

            length = [60, 90, 120, 150, 180]
            for l in length:
                if random.random() < 0.5:
                    continue 

                pair_train_data['%d,%d,%d'%(object_id1,object_id2,l)]= label

                count += 1
                print idx, count

test_list = []
length = [60, 90, 120, 150, 180]
for l in length:
    for idx, (image_id1, image_id2) in enumerate(test_pair_images_id_list):
        if image_id1 not in image_objects or image_id2 not in image_objects:
            continue
        image_objects_ids1 = image_objects.get(image_id1, [])
        image_objects_ids2 = image_objects.get(image_id2, [])
        if image_id1 not in all_images_pathes or image_id2 not in all_images_pathes:
            continue

        random.shuffle(image_objects_ids1)
        for object_id1 in image_objects_ids1:
            same_object_ids2 = [object_id2 for object_id2 in image_objects_ids2 if int(same_objects_id.get('-'.join([str(object_id1), str(object_id2)]), 0) or same_objects_id.get('-'.join([str(object_id2), str(object_id1)]), 0) )]
            if not same_object_ids2:
                continue

            image_objects_ids2.sort(key=lambda object_id2: np.linalg.norm(np.array(objects[object_id2]['bbox']) - np.array(objects[same_object_ids2[0]]['bbox'])))
            near_image_objects_ids2 = image_objects_ids2[0:max_diff_sku_pairs+1]
            for object_id2 in image_objects_ids2:

                object1 = objects[object_id1]
                object2 = objects[object_id2]
                if object1['type'] != object2['type'] and object_id2 not in near_image_objects_ids2:
                    continue
                label = int(object_id2 in same_object_ids2)

                test_list.append(['{} {} {}\n'.format(object_id1, label, l), '{} {} {}\n'.format(object_id2, label, l)])

                count += 1
                print idx, count
            test_list.append(['none none none\n', 'none none none\n'])

JsonTrainPairData = json.dumps(pair_train_data, ensure_ascii=False)

f = open('/data-4t/home/yanjia/siamese_shelf/length_data/json_data/pair_train_data_all_v1.json', 'w')
f.write(JsonTrainPairData)
f.close()

f_test = open('./test/length_image_list_test_all_v1.txt', 'wb')
f_test_p = open('./test/length_image_list_test_all_p_v1.txt', 'wb')
for a, b in test_list:
    f_test.write(a)
    f_test_p.write(b)
f_test.close()
f_test_p.close()
		