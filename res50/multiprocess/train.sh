nohup python tools/train_siamese.py \
--gpu 0 \
--process 4 \
--solver ../merge_models/solver_multiprocess.prototxt  \
--weights ../models/resnet50_v2_iter_220000.caffemodel > train_length_v3.log 2>&1 &
