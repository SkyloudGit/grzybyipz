import os
import tensorflow as tf
import numpy as np
import time
import cv2


class NumberDetector:
    def __init__(self):
        start = time.time()
        tf.keras.backend.clear_session()
        # self.saved_path = os.path.dirname(os.path.abspath(__file__)) + '/resnet_number_detector_2/saved_model'
        self.saved_path = os.getcwd() + '/grzyby_2/saved_model'
        self.label_map = self.read_label_map(os.path.dirname(os.path.abspath(__file__)) + '/label_map.pbtxt')
        self.detection_model = tf.saved_model.load(self.saved_path)
        self.image = None
        self.output_dict = None
        end = time.time()
        # print(end - start, 'init')

    @staticmethod
    def read_label_map(label_map_path):
        item_id = None
        item_name = None
        items = {}

        with open(label_map_path, "r") as file:
            for line in file:
                line.replace(" ", "")
                if line == "item{":
                    pass
                elif line == "}":
                    pass
                elif "id:" in line:
                    item_id = int(line.split(":", 1)[1].strip())
                elif "name" in line:
                    item_name = line.split(":", 1)[1].replace("'", "").strip()

                if item_id is not None and item_name is not None:
                    items[item_id] = item_name
                    item_id = None
                    item_name = None

        return items

    def infer(self, image):
        self.image = image
        rgb_tensor = tf.convert_to_tensor(image, tf.uint8)
        rgb_tensor = tf.expand_dims(rgb_tensor, 0)
        model_fn = self.detection_model.signatures['serving_default']
        self.output_dict = model_fn(rgb_tensor)
        num_detections = int(self.output_dict.pop('num_detections'))
        self.output_dict = {key: value[0, :num_detections].numpy()
                            for key, value in self.output_dict.items()}
        self.output_dict['num_detections'] = num_detections
        self.output_dict['detection_classes'] = self.output_dict['detection_classes'].astype(np.int64)

        return self.output_dict

    def filter_numbers(self, threshold):
        im_height, im_width, _ = self.image.shape
        scores = self.output_dict['detection_scores']
        idx = np.argmax(scores)
        # indexes = np.argwhere((scores > threshold) != 0).flatten()
        # number_names = []
        # number_positions = []
        if scores[idx] > threshold:
            ymin, xmin, ymax, xmax = tuple(self.output_dict['detection_boxes'][idx])
            left, right, top, bottom = (int(val) for val in (xmin * im_width, xmax * im_width,
                                                             ymin * im_height, ymax * im_height))

            self.image = cv2.rectangle(self.image, (left, top), (right, bottom), (0, 255, 0), 4)
            cv2.imwrite('test3.jpg', self.image)

            return self.label_map[self.output_dict['detection_classes'][idx]], str(scores[idx])

        return '', ''
