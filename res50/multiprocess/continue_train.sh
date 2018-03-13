nohup python tools/train_siamese.py \
--gpu 0 \
--process 8 \
-solver /home/zhuyanjia/siamese_shelf/res50_length/continue_solver.prototxt \
-snapshot /home/zhuyanjia/siamese_shelf/res50_length/models/resnet50_length_iter_12000.solverstate \
-gpu 1 > train_length_1.log 2>&1 &