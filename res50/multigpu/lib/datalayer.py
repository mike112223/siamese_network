"""The data layer used during training to train the network.
   Load 2 image at a time and generate all posible bbox pairs
"""

import caffe
import numpy as np
import time
import sys
import os
import cv2

from utils import SampleData_Multi_GPU, FeatureLmdb, cv_load_image, im_list_to_blob
import config
import matplotlib.pyplot as plt

class DataLayerWLoss(caffe.Layer):
    """Sample data layer used for training."""    
    def _get_next_minibatch(self):  
        cnt = 0
        batch_size = self._batch_size
        im_blob = []
        assert self._index_bbox % batch_size == 0, 'error'
        while cnt < batch_size:
            if self._index_bbox >= config.MAX_PAIR:
                self._index_bbox = 0
                self._index_pair += 1
                self._prepare_next_pair()
            im = self._prep_im_with_bbox()
            if im is not None:
                im_blob.append(im)
                cnt += 1
            self._index_bbox += 1
        im_blob = im_list_to_blob(im_blob)
        blobs = {'data': im_blob,
             }
        # print len(blobs['data'])
        # print self._index_bbox, self._bbox_num
        return blobs
        
    def _prep_im_with_bbox(self):
        '''
        crop, mean, resize
        '''
        bbox_idx = self._bbox_permutation[self._index_bbox]
        bbox_a = self._bbox_a[bbox_idx[0]]
        bbox_b = self._bbox_b[bbox_idx[1]]
        if len(bbox_a) == 0 or len(bbox_b) == 0:
            return None
        else:
            bbox_a = [bbox_a[0], bbox_a[1], bbox_a[0]+bbox_a[2], bbox_a[1]+bbox_a[3]]
            bbox_b = [bbox_b[0], bbox_b[1], bbox_b[0]+bbox_b[2], bbox_b[1]+bbox_b[3]]
            im_a = self._img_a[bbox_a[1]:bbox_a[3], bbox_a[0]:bbox_a[2]]
            im_b = self._img_b[bbox_b[1]:bbox_b[3], bbox_b[0]:bbox_b[2]]
            im_a = im_a.astype(np.float32, copy=False)
            im_b = im_b.astype(np.float32, copy=False)
            pixel_means = np.array([[config.PIXEL_MEANS]])
            target_size = config.TARGET_SIZE
            im_a = cv2.resize(im_a, (target_size, target_size),
                        interpolation=cv2.INTER_LINEAR)
            im_b = cv2.resize(im_b, (target_size, target_size),
                        interpolation=cv2.INTER_LINEAR)
            
            ### visurelize training data
            # if bbox_idx[0] == bbox_idx[1]:
            #         label = 1
            #     else:
            #         label = 0
            # if self._iter == 19:
            #self._vis(im_a,im_b)

            im_a -= pixel_means
            im_b -= pixel_means
            im = np.concatenate((im_a, im_b), axis=2)
            return im

    def _vis(self, im_a, im_b, label=1):
        im_a = im_a.astype('uint8')
        im_b = im_b.astype('uint8')
        plt.subplot(1,2,1)
        plt.title(str(label))
        plt.imshow(im_a)
        plt.subplot(1,2,2)
        plt.imshow(im_b)
        plt.show()

    def _load_pair_img(self, pair):
        path_a = self._data_container._imgid_pairs[pair]['path_a']
        img_a = cv_load_image(path_a)
        path_b = self._data_container._imgid_pairs[pair]['path_b']
        img_b = cv_load_image(path_b)
        return img_a, img_b

    def _prepare_next_pair(self):
        '''
        read the next image pair and shuffle the bboxes
        '''
        # print 'next pair'
        if self._index_pair >= len(self._data_container._keys):
            self._index_pair = 0
            np.random.shuffle(self._data_container._keys)
        pair = self._data_container._keys[self._index_pair]
        self._img_a, self._img_b = self._load_pair_img(pair)
        bbox_a_pre = self._data_container._imgid_pairs[pair]['bbox_a']
        bbox_b_pre = self._data_container._imgid_pairs[pair]['bbox_b']
        assert len(bbox_a_pre) == len(bbox_b_pre) 
        bbox_a = []
        bbox_b = []
        for idx in range(len(bbox_a_pre)):
            if len(bbox_a_pre[idx]) != 0 and len(bbox_b_pre[idx]) != 0:
                bbox_a.append(bbox_a_pre[idx])
                bbox_b.append(bbox_b_pre[idx])
        assert len(bbox_a) >= config.MAX_PAIR, len(bbox_a)
        idx = [i for i in range(len(bbox_a))]
        np.random.shuffle(idx)
        idx = idx[:config.MAX_PAIR]
        self._bbox_a = []
        self._bbox_b = []
        for i in idx:
            self._bbox_a.append(bbox_a[i])
            self._bbox_b.append(bbox_b[i])
        self._bbox_permutation = []
        for i in range(len(self._bbox_a)):
            self._bbox_permutation.append([i,i])
        # print len(positive_pair), len(negative_pair)

    def set_roidb(self, gpu_id=0):
        # print "set_roidb ****************"
        self._batch_size = config.BATCH_SIZE
        self._data_container = SampleData_Multi_GPU()               
        self._index_pair = 0
        self._index_bbox = 0
        np.random.seed(gpu_id)
        np.random.shuffle(self._data_container._keys)
        pair = self._data_container._keys[0]
        self._prepare_next_pair()
        self._iter = 0
     
    def setup(self, bottom, top):
        """Setup the RoIDataLayer.""" 
        # print "setup *****************"
        batch_size = config.BATCH_SIZE   
        self._name_to_top_map = {
            'data': 0,
            }
        top[0].reshape(batch_size, 6, 224, 224)             
             
    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        # print self._iter, 'start'
        # print "forward ***********"
        blobs = self._get_next_minibatch()
        # print self._iter, self._index_pair
        self._iter += 1
        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            top[top_ind].data[...] = blob

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

if __name__ == '__main__':
    pass
