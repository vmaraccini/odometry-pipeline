import os
import tarfile

import numpy as np
from PIL import Image

import tensorflow as tf
from extraction.extractor import FeatureExtractor
import cv2

colormap = np.zeros((256, 3), dtype=int)
ind = np.arange(256, dtype=int)


def bit_get(val, idx):
    return (val >> idx) & 1


for shift in reversed(range(8)):
    for channel in range(3):
        colormap[:, channel] |= bit_get(ind, channel) << shift
    ind >>= 3


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]

        resized = cv2.resize(np.array(seg_map).astype(np.uint8), (width, height))

        dilated = cv2.dilate(resized, np.ones((5, 5), np.uint8), iterations=1)
        return dilated


class DeepExtractor:
    model = None
    extractor = FeatureExtractor()

    def __init__(self, model_path="model/deeplab_model.tar.gz"):
        self.model = DeepLabModel(model_path)

    def extract(self, frame):
        candidates = self.extractor.extract(frame)
        segmented = np.array(self.model.run(Image.fromarray(frame)))

        # background = 0
        # aeroplane
        # bicycle
        # bird
        # boat
        # bottle
        # bus
        # car
        # cat
        # chair
        # cow
        # diningtable
        # dog
        # horse
        # motorbike
        # person
        # pottedplant
        # sheep
        # sofa
        # train
        # tvmonitor

        non_moving_labels = [0]

        filtered = [kp for kp in candidates if segmented[int(kp[1]), int(kp[0])] in non_moving_labels]

        display = frame.copy()
        [cv2.circle(display, tuple(pt), 2, color=(0, 255, 0)) for pt in filtered]
        cv2.imshow('Filtered', display)

        cv2.waitKey(1)
        cv2.waitKey(1)

        return np.array(filtered)
