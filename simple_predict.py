import os
import urllib.request
import cv2
import imageio
import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell
from skimage import measure
import numpy as np
import onnxruntime as ort

ope = os.path.exists
opj = os.path.join

PRETRAINED_DIR = './data'
NUC_MODEL = f'{PRETRAINED_DIR}/nuclei-model.pth'
CELL_MODEL = f'{PRETRAINED_DIR}/cell-model.pth'
COLORS =  ["red", "green", "blue", "yellow"]
LABELS = {
  0: 'Nucleoplasm',
  1: 'Nuclear membrane',
  2: 'Nucleoli',
  3: 'Nucleoli fibrillar center',
  4: 'Nuclear speckles',
  5: 'Nuclear bodies',
  6: 'Endoplasmic reticulum',
  7: 'Golgi apparatus',
  8: 'Intermediate filaments',
  9: 'Actin filaments',
  10: 'Microtubules',
  11: 'Mitotic spindle',
  12: 'Centrosome',
  13: 'Plasma membrane',
  14: 'Mitochondria',
  15: 'Aggresome',
  16: 'Cytosol',
  17: 'Vesicles and punctate cytosolic patterns',
  18: 'Negative',
}

def load_rgby(image_dir, image_id, suffix='jpg', in_channels=4):
  images = [
    cv2.imread(f'{image_dir}/{image_id}_{color}.{suffix}', cv2.IMREAD_GRAYSCALE) for color in COLORS[0:in_channels]
  ]
  for image in images:
    if image is None:
      return None
  rgby = np.stack(images, axis=-1)
  return rgby

def cell_crop_augment(image, mask, paddings=(20, 20, 20, 20)):
  top, bottom, left, right = paddings
  label_image = measure.label(mask)
  max_area = 0
  for region in measure.regionprops(label_image):
    if region.area > max_area:
      max_area = region.area
      min_row, min_col, max_row, max_col = region.bbox

  min_row, min_col = max(min_row - top, 0), max(min_col - left, 0)
  max_row, max_col = min(max_row + bottom, mask.shape[0]), min(max_col + right, mask.shape[1])

  image = image[min_row:max_row, min_col:max_col]
  mask = mask[min_row:max_row, min_col:max_col]
  return image, mask

def load_mask(cellmask_dir, image_id, masktype='cellmask'):
  mask = cv2.imread(f'{cellmask_dir}/{image_id}_{masktype}.png', flags=cv2.IMREAD_GRAYSCALE)
  return mask

def generate_cell_indices(cell_mask):
  cell_indices = np.sort(list(set(np.unique(cell_mask).tolist()) - {0, }))
  return cell_indices

def fetch_image(work_dir, img_id):
    v18_url = 'http://v18.proteinatlas.org/images/'
    img_id_list = img_id.split('_')
    for color in COLORS:
        img_url = v18_url + img_id_list[0] + '/' + '_'.join(img_id_list[1:]) + '_' + color + '.jpg'
        img_name = img_id + '_' + color + '.png'
        fpath = os.path.join(work_dir, img_name)
        if not os.path.exists(fpath):
            urllib.request.urlretrieve(img_url, opj(work_dir, '_.jpg'))
            image = cv2.imread(opj(work_dir, '_.jpg'), flags=cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(fpath, image)

def crop_image(image_raw, mask_raw, image_id, image_size, save_dir=None, save_rgby=False):
  
  if (image_size > 0) & (image_raw.shape[0] != image_size):
    image_raw = cv2.resize(image_raw, (image_size, image_size))
  
  if (image_size > 0) & (mask_raw.shape[0] != image_size):
    mask_raw = cv2.resize(mask_raw, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
  cell_indices = generate_cell_indices(mask_raw)
  crop_images = []
  for maskid in cell_indices:
    image = image_raw.copy()
    mask = mask_raw.copy()
   
    image[mask != maskid] = 0
    image, _ = cell_crop_augment(image, (mask == maskid).astype('uint8'))
    if save_dir:
      save_fname = f'{save_dir}/{image_id}_{maskid}.png'
      cv2.imwrite(save_fname, image if save_rgby else image[:, :, :3]) # ignore the alpha channel
    crop_images.append(image)
  return cell_indices, crop_images


def load_image_mask_labels(image_dir, image_id, maskid, image_size=128):
    image = cv2.imread(f'{image_dir}/{image_id}_{maskid}.png', flags=cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (image_size, image_size))
    assert image.shape == (image_size, image_size, 4), image.shape
    return image


image_ids = ['115_672_E2_1']
work_dir = './data'
device = 'cuda:2'
threshold=0.3
image_size = 128
model_path = os.path.join(work_dir, 'bestfitting-inceptionv3-single-cell.onnx')

os.makedirs(work_dir, exist_ok=True)
if not os.path.exists(model_path):
    print('Downloading bestfitting-inceptionv3-single-cell.onnx...')
    urllib.request.urlretrieve('https://github.com/oeway/hpa-bestfitting-inceptionv3/releases/download/v0.1.0/bestfitting-inceptionv3-single-cell.onnx', model_path)

for image_id in image_ids:
    fetch_image(work_dir, image_id)

segmentator = cellsegmentator.CellSegmentator(
  NUC_MODEL,
  CELL_MODEL,
  scale_factor=0.25,
  device=device,
  padding=True,
  multi_channel_model=True,
)

ort_session = ort.InferenceSession(model_path)
crops_dir = opj(work_dir, 'crops')
seg_dir = opj(work_dir, 'segmentations')
os.makedirs(crops_dir, exist_ok=True)
os.makedirs(seg_dir, exist_ok=True)
for image_id in image_ids:
    images = [[f'{work_dir}/{image_id}_red.png'], [f'{work_dir}/{image_id}_yellow.png'], [f'{work_dir}/{image_id}_blue.png']]
    nuc_segmentations = segmentator.pred_nuclei(images[2])
    cell_segmentations = segmentator.pred_cells(images)
    nuclei_mask, cell_mask = label_cell(nuc_segmentations[0], cell_segmentations[0])
    if seg_dir:
      imageio.imwrite(f'{seg_dir}/{image_id}_cell.png', cell_mask)
      imageio.imwrite(f'{seg_dir}/{image_id}_nuclei.png', nuclei_mask)
    image_raw = load_rgby(work_dir, image_id, suffix='png')
    cell_indices, crop_images = crop_image(image_raw, cell_mask, image_id, image_size, crops_dir, False)
    for mask_id, image in zip(cell_indices, crop_images):
        image = cv2.resize(image, (image_size, image_size))
        image = image.transpose(2, 0, 1)  # HxWxC to CxHxW
        image = image / 255.
        image = image[None, :, :, :]
        classes, features = ort_session.run(None, {'image': image.astype(np.float32)})
        preds = [(LABELS[i], prob) for i, prob in enumerate(classes[0].tolist()) if prob>threshold]
        print(image_id, mask_id, preds, features.shape)