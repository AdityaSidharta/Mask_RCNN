import os
from mrcnn.utils import Dataset
from skimage.color import gray2rgb
import pandas as pd
import pydicom
import numpy as np


class PneumoniaDataset(Dataset):
    def __init__(self, data_dir, mode):
        assert mode in ['dev', 'val', 'train', 'test']

        super().__init__(self)
        self.data_dir = data_dir
        self.mode = mode
        self.img_path = os.path.join(self.data_dir, 'stage_1_{}_images'.format(self.mode))
        self.train_labels_df = pd.read_csv(os.path.join(self.data_dir, 'stage_1_train_labels.csv'))
        self.train_labels_dict = self.create_train_labels_dict()

        self.add_class('Pneumonia', 1, 'Pneumonia')
        self.add_all_images()

    def add_all_images(self):
        img_fn_list = [x for x in os.listdir(self.img_path) if x.endswith('.dcm')]
        for idx, img_fn in enumerate(img_fn_list):
            patient_id = img_fn.split('.')[0]
            self.add_image(source="Pneumonia",
                           image_id=idx,
                           path=os.path.join(self.img_path, img_fn),
                           patient_id=patient_id)

    def create_train_labels_dict(self):
        train_labels_dict = {}
        for key, row in self.train_labels_df.iterrows():
            if row.patientId not in train_labels_dict.keys():
                train_labels_dict[row.patientId] = [{
                    'x': row.x,
                    'y': row.y,
                    'width': row.width,
                    'height': row.height,
                    'Target': row.Target
                }]
            else:
                train_labels_dict[row.patientId].append(
                    {
                        'x': row.x,
                        'y': row.y,
                        'width': row.width,
                        'height': row.height,
                        'Target': row.Target
                    }
                )
        return train_labels_dict

    def image_reference(self, image_id):
        return self.image_info[image_id]['path']

    def load_image(self, image_id):
        path = self.image_info[image_id]['path']
        img = pydicom.read_file(path).pixel_array
        img = gray2rgb(img)
        return img

    def load_mask(self, image_id):
        patient_id = self.image_info[image_id]['patient_id']
        full_pdict = self.train_labels_dict[patient_id]
        n_mask = len(full_pdict)
        mask = np.zeros((1024, 1024, n_mask))
        class_mask = np.zeros(n_mask)
        for idx_mask, pdict in enumerate(full_pdict):
            target = int(pdict['Target'])
            if target == 1:
                x, y, width, height = \
                    int(pdict['x']), int(pdict['y']), int(pdict['width']), int(pdict['height'])
                mask[y:(y+height), x:(x+width), idx_mask] = 1.
                class_mask[idx_mask] = 1.
        return mask.astype(np.bool), class_mask.astype(np.int32)






