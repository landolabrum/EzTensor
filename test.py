# /home/deepturn/tensorflow/object_detection/test.py



# Configure the system paths. You can do it with the following command every time you start a new terminal after activating your conda environment.


# CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
# For your convenience it is recommended that you automate it with the following commands. The system paths will be automatically configured when you activate this conda environment.


# mkdir -p $CONDA_PREFIX/etc/conda/activate.d
# echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh




import os
import numpy as np
from PIL import Image
import six.moves.urllib as urllib
from six import BytesIO
import tarfile
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

import matplotlib.pyplot as plt
# Define the URLs for model and labels
MODEL_DOWNLOAD_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_512x512_coco17_tpu-8.tar.gz'
LABELS_DOWNLOAD_URL = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt'

# Define the local file paths for model and labels
LOCAL_MODEL_DIR = 'models'
LOCAL_LABELS_PATH = os.path.join('data', 'mscoco_label_map.pbtxt')




# Ensure the directories exist
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOCAL_LABELS_PATH), exist_ok=True)

# Download the model if it doesn't exist
if not os.path.exists(os.path.join(LOCAL_MODEL_DIR, 'centernet_hg104_512x512_coco17_tpu-8', 'saved_model')):
    opener = urllib.request.URLopener()
    opener.retrieve(MODEL_DOWNLOAD_URL, 'centernet_hg104_512x512_coco17_tpu-8.tar.gz')
    tar_file = tarfile.open('centernet_hg104_512x512_coco17_tpu-8.tar.gz')
    tar_file.extractall(LOCAL_MODEL_DIR)
    tar_file.close()
    os.remove('centernet_hg104_512x512_coco17_tpu-8.tar.gz')

# Download the labels if they don't exist
if not tf.io.gfile.exists(LOCAL_LABELS_PATH):
    urllib.request.urlretrieve(LABELS_DOWNLOAD_URL, LOCAL_LABELS_PATH)

# Load the model
model = tf.saved_model.load(os.path.join(LOCAL_MODEL_DIR, 'centernet_hg104_512x512_coco17_tpu-8', 'saved_model'))

# Load label map data for COCO
category_index = label_map_util.create_category_index_from_labelmap(LOCAL_LABELS_PATH, use_display_name=True)

# Function to load an image into a numpy array
def load_image_into_numpy_array(path):
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# Function to run inference on an image
def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)
    return output_dict

# Function to run the detection process
def detect(input_image_path, output_image_path):
    image_np = load_image_into_numpy_array(input_image_path)
    output_dict = run_inference_for_single_image(model, image_np)
    
    # Extract the labels for each detection
    detection_classes = output_dict['detection_classes'][0].numpy().astype(np.int32)
    
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'][0].numpy(),
        detection_classes,
        output_dict['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)
    
    # Create a figure to save the output image (This is commented out since it's not needed)
    plt.figure(figsize=(24,32))
    # plt.imshow(image_np)
    plt.savefig(output_image_path)

# Define directories for input and output images
input_images_dir = "test_images"
output_images_dir = "output_images"

# Get the list of input image paths
input_image_paths = [os.path.join(input_images_dir, name) for name in os.listdir(input_images_dir) if name.endswith(".jpg")]

# Run the detection and save the output images
for input_image_path in input_image_paths:
    print(f"[ FOUND IMG ]: {input_image_path}")
    output_image_path = os.path.join(output_images_dir, os.path.basename(input_image_path))
    detect(input_image_path, output_image_path)
