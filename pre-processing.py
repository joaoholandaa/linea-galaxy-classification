import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import operator

main_path = "./data/*"
name_fits = str
hog_img = str

#EMPTY ARRAY LIST
log_list=[]
crop_list=[]
norm_list=[]

#FUNCTIONS
def fits_work_log(name_fits):
  image_file = fits.open(name_fits)
  image_data = image_file[0].data
  c = 255 / np.log(1 + np.max(image_data))
  log_image = c * (np.log(image_data + 1))
  log_image = np.array(log_image, dtype = np.uint8)
  log_list.append(log_image)

def cropND(img, bounding):
  start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
  end = tuple(map(operator.add, start, bounding))
  slices = tuple(map(slice, start, end))
  return img[slices]

def norm(log_img):
  norm = np.linalg.norm(log_img)
  normalized_array_log_image = log_img/norm
  return normalized_array_log_image

#INTERACTIONS
for i in glob.glob(main_path):
  print(i)
  fits_work_log(i)

for i in range(len(log_list)):
  log_crop_img = log_list[i]
  new_log_crop_img = cropND(log_crop_img, (50,50))
  crop_list.append(new_log_crop_img)

for i in range(len(crop_list)):
  norm_log_crop_img = crop_list[i]
  norm_log_array = norm(norm_log_crop_img)
  norm_list.append(norm_log_array)

plt.imshow(norm_list[0], cmap='gray')
plt.show()
plt.imshow(norm_list[1], cmap='gray')
plt.show()
