from __future__ import division
from PIL import Image
from PIL import ImageEnhance
import numpy as np
import torch
def random_brightness(img, factor):
	"""random change the brightness of an img"""
	enhancer = ImageEnhance.Brightness(img)
	selector = np.random.uniform(low=factor)
	return enhancer.enhance(selector)
def random_contrast(img, factor):
	"""random change the contrast of an img"""
	enhancer = ImageEnhance.Contrast(img)
	selector = np.random.uniform(low=factor)
	return enhancer.enhance(factor)
def random_scale(img, min_size=256, max_size=448):
	"""random scale img with shorter side"""
	target = np.random.randint(min_size, max_size+1)
	target_h, target_w = target, target
	w,h = img.size
	
	if w>h:
		target_w = int(np.round(target * w/h))
	else:
		target_h = int(np.round(target * h/w))

	img = img.resize((target_w, target_h))
	return img

def random_hflip(img):
	"""randomly horizen flip an img"""
	selector = np.random.uniform()
	if selector > 0.5:
		return img.transpose(Image.FLIP_LEFT_RIGHT)
	else:
		return img

def random_vflip(img):
	"""randomly vertical flip an img"""
	selector = np.random.uniform()
	if selector > 0.5:
		return img.transpose(Image.FLIP_TOP_BOTTOM)
	else:
		return img

def random_transpose(img):
	"""randomly transpose an img with main diag"""
	selector = np.random.uniform()
	if selector > 0.5:
		return img.transpose(Image.TRANSPOSE)
	else:
		return img

def random_transpose_(img):
	"""randomly transpose an img with diag"""
	selector = np.random.uniform()
	if selector>0:
		w,h = img.size
		img_data = np.array(img.getdata()).astype(np.uint8).reshape((h,w,3))
		transposed_img = np.zeros((w,h,3)).astype(np.uint8)
		start = (0, w-1)
		for c in range(3):
			for t in range(min(w,h)):
				for i in range(w-t):
					transposed_img[i+t,h-1-t,c] = img_data[t,start[1]-t-i,c]
				for j in range(h-t):
					transposed_img[t,h-1-t-j,c] = img_data[t+j,start[1]-t,c]

		return Image.fromarray(transposed_img)
	else:
		return img 

def random_rotate(img):
	"""randomly rotate an img with angle 90 180 270"""
	selector = np.random.randint(4)
	if selector == 0:
		return img.transpose(Image.ROTATE_90)
	elif selector == 1:
		return img.transpose(Image.ROTATE_180)
	elif selector == 2:
		return img.transpose(Image.ROTATE_270)
	else:
		return img

def random_crop(img, size=224):
	"""random crop an img with size"""
	w,h = img.size
	left = np.random.randint(w-size)
	upper = np.random.randint(h-size)
	return img.crop((left, upper, left+size, upper+size))

def to_tensor(img):
	img_data = np.array(img, np.int32, copy=False)
	img_tensor = torch.from_numpy(img_data.transpose((2,0,1)))
	return img_tensor.float().div(255)

def normalize(img, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]):
    for t, m, s in zip(img, mean, std):
        t.sub_(m).div_(s)
    return img

if __name__ == '__main__':
	img = Image.open('/home/yy/cat.jpg').resize((188,188))
	# img.show()
	img.transpose(Image.ROTATE_270).show()
	# img.transpose(Image.TRANSPOSE).show()
	random_transpose_(img).show()
