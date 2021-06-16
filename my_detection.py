import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import numpy as np
import cv2

from six import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
import pytesseract

pytesseract.pytesseract.tesseract_cmd = './Tesseract-OCR/tesseract'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
MODEL_DIR = 'trained-inference-graphs\output'
PATH_TO_CKPT = os.path.join(MODEL_DIR,'checkpoint')
PATH_TO_CFG = os.path.join(MODEL_DIR, 'pipeline.config')
LABEL_MAP_PATH = 'label_map.pbtxt'
CATEGORY_INDEX = label_map_util\
.create_category_index_from_labelmap(LABEL_MAP_PATH,
                                     use_display_name=True)

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(
      model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])
def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    image = np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
    if (im_width, im_height)!=(640,640):
        image = cv2.resize(image, (640,640))
    return image
def get_abs_box_coords(image,
               boxes,
               classes,
               scores,
               category_index,
               max_boxes_to_draw=1,
               min_score_thresh=.02):
    img_height, img_width, img_channel = image.shape

    box_dict = {'detection_classes':[], 'detection_boxes':[], 'detection_scores':[]}

    for i in range(max_boxes_to_draw):
        if scores[i] < min_score_thresh:
            continue
        box = boxes[i]
        ymin, xmin, ymax, xmax = box
        x_up = int(xmin*img_width)
        y_up = int(ymin*img_height)
        x_down = int(xmax*img_width)
        y_down = int(ymax*img_height)
        box = np.array([x_up,y_up,x_down,y_down])
        class_ = category_index[classes[i]]['name']
        score = scores[i]

        box_dict['detection_scores'].append(score)
        box_dict['detection_boxes'].append(box)#xmin,ymin,xmax,ymax
        box_dict['detection_classes'].append(class_)
    return box_dict

def image_to_text(image_path):

    image_np = load_image_into_numpy_array(image_path)
    input_tensor = tf.convert_to_tensor( np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)
    boxes = get_abs_box_coords(
          image_np,
          detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy() + 1).astype(int),
          detections['detection_scores'][0].numpy(),
          CATEGORY_INDEX,

          max_boxes_to_draw=1,
          min_score_thresh=.01
        )
    for i in range(len(boxes['detection_boxes'])):
        crop = boxes['detection_boxes'][i]
        crop_img = Image.fromarray(image_np).crop((crop))
        text = pytesseract.image_to_data(crop_img,config='--psm 13 --oem 3',
                                            output_type=pytesseract.Output.DICT,
                                            lang='eng')['text'][-1]
        print(text)
#         plt.imshow(crop_img)
#         plt.show()
    return text


if __name__=='__main__':
    image_path = os.path.join('', '883.jpg')
    image_to_text(image_path)
