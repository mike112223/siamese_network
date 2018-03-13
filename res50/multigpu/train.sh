nohup python -u tools/train_multi_gpus.py \
--gpu 0,1 \
--solver ../merge_models/solver_multigpu.prototxt \
--weights ../merge_models/siamese_res50.caffemodel > train_verify.log 2>&1 &
