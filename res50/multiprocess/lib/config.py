
USE_LMDB = False
IMG_LMDB_PATH = '/data/data/liuhuawei/data_lmdb_backup_for_ssd/data_lmdb_for_image_copy_and_mark_data'

METADATA_JSON = '/data-4t/home/yanjia/siamese_shelf/length_data/json_data/objectid_to_metadata.json'

## json path to pair file, key:(objectid_a, objectid_b) val:1 or 0
PAIR_JSON = '/data-4t/home/yanjia/siamese_shelf/length_data/json_data/pair_train_data_shelf_9_10_11.json'
## image config
TARGET_SIZE = 224
PIXEL_MEANS = [104.0, 117.0, 123.0]

## The number of samples in each minibatch
BATCH_SIZE = 8

## prefetch process for data layer (must be false here)
USE_PREFETCH = False
RNG_SEED = 8

# BBOX_SCALE_TYPE = 'SCALE'
BBOX_SCALE_TYPE = 'ABSOLUTE'
BBOX_SCALE = 3.0
VIS = False
