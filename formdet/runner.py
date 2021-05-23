import streamlit as st
from PIL import Image
import numpy as np
import os
from pathlib import Path

import os



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2', '3'}
import tensorflow as tf

tf.autograph.set_verbosity(1)
tf.get_logger().setLevel('ERROR')

from collections import defaultdict
from io import StringIO

import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display


from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

import time
from object_detection.utils import config_util
from object_detection.builders import model_builder

from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow, subplots
from matplotlib.colors import to_hex

tf.debugging.set_log_device_placement(True)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


st.set_page_config(page_title="Form components detection", layout="wide", page_icon="🛰",
                  # initial_sidebar_state="collapsed"
                   )

# model_dir = Path('/home/ubuntu/formsdet/.trained_models/')
working_dir = Path.cwd()
print(f'working dir: {str(working_dir)}')
model_dir = Path('/root/objectdet-check/')

labels_file = model_dir / 'label_map2.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(labels_file, use_display_name=True)

type_to_color = {
    8: 'black',
    1:'green',
    6: 'mediumblue',
    4:'magenta',
    7:'yellow',
    2:'gold',
    3:'dodgerblue',
    5:'darkviolet',
}

class ScoredRectangle:
    def __init__(self, b, score=None, label=None):
        self.x1 = b[1]
        self.y1 = b[0]
        self.x2 = b[3]
        self.y2 = b[2]
        self.score = score
        self.label = label

    def is_intersect(self, other):
        if self.x1 > other.x2 or self.x2 < other.x1:
            return False
        if self.y1 > other.y2 or self.y2 < other.y1:
            return False
        return True
    
    def intersect(self, r):        
        x1 = max(self.x1, r.x1)
        y1 = max(self.y1, r.y1)
        x2 = min(self.x2, r.x2)
        y2 = min(self.y2, r.y2)
        
        return ScoredRectangle([y1,x1,y2,x2])
    
    def area(self,):
        return (self.x2-self.x1)*(self.y2-self.y1)
    
    def size(self, ):
        return (self.x2-self.x1, self.y2-self.y1)
    
    def mul(self, page_width, page_height):
        self.x1 = self.x1 * page_width
        self.x2 = self.x2 * page_width
        self.y1 = self.y1 * page_height
        self.y2 = self.y2 * page_height
        
    def __str__(self):
        return f'x1:{self.x1} x2:{self.x2} y1:{self.y1} y2:{self.y2} size:{self.size()}'
        
        
def get_iou_boxes(r1: ScoredRectangle, r2: ScoredRectangle):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Returns
    -------
    float
        in [0, 1]
    """
#     r1 = Rectangle(b1)
#     r2 = Rectangle(b2)
    
    rr = r1.intersect(r2)
    
    if rr.x2 < rr.x1 or rr.y2 < rr.y1:
        return 0.0
    
    a1 = r1.area()
    a2 = r2.area()
    ra = float(rr.area())
#     iou = ra / float(a1+a2-ra)
    iou = max(ra/a1, ra/a2)
    return iou
 

@st.cache(allow_output_mutation=True, show_spinner=True)
def get_model():
    pipeline_path = model_dir / "ssd_efficientdet_d4_1024x1024_coco17_tpu-32.config"
    # checkpoint_path = model_dir / "checkpoint"

    print('Loading model... ', end='')

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(str(pipeline_path))
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    # ckpt.restore(str(model_dir / '.checkpoints' / 'ckpt-41')).expect_partial()
    ckpt.restore(str(model_dir / '.checkpoints.bd' / 'ckpt-101')).expect_partial()
    
    return detection_model

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

detection_model = get_model()

def draw_detections(ax, pillow_image, threshold=0.15, print_annotation=True, lowest_threshold=.1, iou_threshold=1.):
    image = pillow_image
    width, height = image.size 
    
    image_np = np.array(image)
 
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
#     print(f'num detections:{num_detections}')

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    ax.set_title(f'processed (num detections={num_detections}, score threshold={lowest_threshold}, iou threshold={iou_threshold}): ')
    imp = ax.imshow(image)
    ax.axis('on')

    list_boxes = []
#     lowest_threshold = .1
    for box, class_i, score in zip(detections['detection_boxes'], detections['detection_classes']+label_id_offset, detections['detection_scores']):
        if score > lowest_threshold:
            r = ScoredRectangle(box, score=score, label=class_i)
            r.mul(width, height)
            list_boxes.append(r)
    
    print(f'total boxes:{len(list_boxes)}')

    
    filtered_boxes = []
    list_boxes.reverse()
    for i, r1 in enumerate(list_boxes):

        for r2 in list_boxes[i+1:]:
            iou = get_iou_boxes(r1, r2)
            if iou > iou_threshold:
                if r1.score < r2.score:
                    break
        else:
            filtered_boxes.append(r1)
        
        
    for r in filtered_boxes:
        x = r.x1
        y = r.y1
        w, h = r.size()
    
        ax.add_patch(Rectangle((x, y), w, h, fill=False, color=type_to_color[r.label], linewidth=2, 
#             hatch='//'
        ))
        if print_annotation:
            ax.text(
                # x+0.5*w, y+0.5*h, 
                x+5, y+0.5*h, 
                f"{category_index[r.label]['name']}:{r.score:.3}", 
                ha='left', va='center', 
                fontsize=9, 
                color=type_to_color[r.label], 
                # bbox=dict(color='white',facecolor='white', alpha=0.95)
            )
    


st.title('Form fields detector')
lowest_threshold = st.sidebar.number_input('lowest box score threshold:', min_value=0., max_value=1., value=0.08, step=0.01)
iou_threshold = st.sidebar.number_input('area intersection ratio threshold:', min_value=0., max_value=1., value=0.2, step=0.01)
show_labels = st.sidebar.checkbox(' show labels', value=True)

uploaded_image_file = st.file_uploader('Download page:', type=["png", "jpg", "jpeg", "tif", "tiff"])
if uploaded_image_file:
    with st.spinner("Prediction..."):
        # Using PIL
        image = Image.open(uploaded_image_file)
#         img_arr = np.array(image)
        fig, axs = plt.subplots(1, 2, figsize=(20, 20), sharey=True)
        axs[0].set_title('original:')
        axs[0].imshow(image, interpolation='lanczos')
        draw_detections(axs[1], image, lowest_threshold=lowest_threshold, iou_threshold=iou_threshold, print_annotation=show_labels)  
        fig.tight_layout(pad=1.)
        st.pyplot(fig)