import os
import collections
import tempfile
import urllib
import tensorflow as tf

_MODEL_URLS = {
    'xception_coco_voctrainaug': 'http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    'xception_coco_voctrainval': 'http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz',
}

Config = collections.namedtuple('Config', 'model_url, model_dir')


def get_config(model_name, model_dir):
    return Config(_MODEL_URLS[model_name], model_dir)


config = get_config(model_name=list(_MODEL_URLS.keys())[0], model_dir='model')

# Check configuration and download the model

_TARBALL_NAME = 'deeplab_model.tar.gz'

model_dir = config.model_dir or tempfile.mkdtemp()
tf.gfile.MakeDirs(model_dir)

download_path = os.path.join(model_dir, _TARBALL_NAME)

if not os.path.isfile(download_path):
    print('downloading model to %s, this might take a while...' % download_path)
    urllib.request.urlretrieve(config.model_url, download_path)
    print('download completed!')
else:
    print('Model file found!')
