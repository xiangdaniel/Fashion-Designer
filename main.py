import os
import random
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import models.mrcnn.coco as coco
import models.mrcnn.utils as utils
import models.mrcnn.model as modellib
import models.mrcnn.visualize as visualize
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import glob
import torch


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

# Directory of videos to run detection
VIDEO_DIR = os.path.join(ROOT_DIR, "videos")
VIDEO_SAVE_DIR = os.path.join(VIDEO_DIR, "save")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object
model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
if config.GPU_COUNT:
    model = model.cuda()

# Load weights trained on MS-COCO
model.load_state_dict(torch.load(COCO_MODEL_PATH))

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def make_video(outvid, images=None, fps=30, size=None, is_color=True, format="FMP4"):
    """
    Create a video from a list of images.

    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
    """

    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid


def main(batch_size, fps, num_change):
    fashion_dir = os.path.join(ROOT_DIR, "results_fashion")
    file_names = next(os.walk(fashion_dir))[2]
    fashion_name = os.path.join(fashion_dir, random.choice(file_names))

    capture = cv2.VideoCapture(os.path.join(VIDEO_DIR, 'IMG_0751.MOV'))
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        fps1 = capture.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps1))
    else:
        fps1 = capture.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps1))
    #capture.release()

    # Playing video from file:
    #capture.set(cv2.CAP_PROP_FPS, fps)

    try:
        os.makedirs(VIDEO_SAVE_DIR)
    except OSError:
        pass

    frame_count = 0
    frames = []
    while True:
        if frame_count % num_change == 0:
            fashion_name = os.path.join(fashion_dir, random.choice(file_names))

        ret, frame = capture.read()

        # Bail out when the video file ends
        if not ret:
            break

        # Save each frame of the video to a list
        frame_count += 1
        frames.append(frame)
        print('frame_count %d' % frame_count)
        if len(frames) == batch_size:
            results = model.detect(frames)
            for i, item in enumerate(zip(frames, results)):
                frame = item[0]
                r = item[1]
                frame = visualize.display_instances(
                    frame, r['rois'], r['masks'], r['class_ids'], class_names, fashion_name, r['scores']
                )
                plt.show()
                name = '{0}.jpg'.format(frame_count + i - batch_size)
                plt.savefig(os.path.join(VIDEO_SAVE_DIR, name))
                #name = os.path.join(VIDEO_SAVE_DIR, name)
                #cv2.imwrite(name, frame)
            # Clear the frames array to start the next batch
            frames = []

    # Get all image file paths to a list.
    images = list(glob.iglob(os.path.join(VIDEO_SAVE_DIR, '*.*')))
    
    # Sort the images by name index.
    images = sorted(images, key=lambda x: float(os.path.split(x)[1][:-3]))

    outvid = os.path.join(VIDEO_DIR, "output.mp4")
    make_video(outvid, images, fps=fps)


if __name__ == '__main__':
    batch_size = 1
    fps = 30
    num_change = 5
    main(batch_size, fps, num_change)
