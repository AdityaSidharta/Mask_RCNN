import os

project_path = os.getenv('PROJECT_PATH')
data_dir = os.path.join(project_path, 'data')
logs_dir = os.path.join(project_path, 'logs')
coco_weights_path = os.path.join(project_path, 'mask_rcnn_coco.h5')
output_path = os.path.join(project_path, 'output')