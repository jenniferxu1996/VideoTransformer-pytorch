# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Copy-paste from DINO library:
https://github.com/facebookresearch/dino
"""
import os
import argparse
import cv2
import random
import colorsys
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image
from skimage.measure import find_contours
from matplotlib.patches import Polygon
from torch.utils.data import DataLoader

import utils
from video_transformer import TimeSformer, ViViT
import data_transform as T
from dataset import DecordInit
import weight_init

# matplotlib.use('Agg')

company_colors = [
	(0,160,215), # blue
	(220,55,60), # red
	(245,180,0), # yellow
	(10,120,190), # navy
	(40,150,100), # green
	(135,75,145), # purple
]
company_colors = [(float(c[0]) / 255.0, float(c[1]) / 255.0, float(c[2]) / 255.0) for c in company_colors]

def apply_mask2(image, mask, color, alpha=0.5):
	"""Apply the given mask to the image.
	"""
	t= 0.2
	mi = np.min(mask)
	ma = np.max(mask)
	mask = (mask - mi) / (ma - mi)
	for c in range(3):
		image[:, :, c] = image[:, :, c] * (1 - alpha * np.sqrt(mask) * (mask>t))+ alpha * np.sqrt(mask) * (mask>t) * color[c] * 255
	return image

def random_colors(N, bright=True):
	"""
	Generate random colors.
	"""
	brightness = 1.0 if bright else 0.7
	hsv = [(i / N, 1, brightness) for i in range(N)]
	colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
	random.shuffle(colors)
	return colors

def show_attn(img, attentions, w_featmap, h_featmap, frame_index, index=None):

	nh = attentions.shape[0] # number of head

	# we keep only the output patch attention
	attentions = attentions.reshape(nh, -1)

	if args.threshold is not None:
		# we keep only a certain percentage of the mass
		val, idx = torch.sort(attentions)
		val /= torch.sum(val, dim=1, keepdim=True)
		cumval = torch.cumsum(val, dim=1)
		th_attn = cumval > (1 - args.threshold)
		idx2 = torch.argsort(idx)
		for head in range(nh):
			th_attn[head] = th_attn[head][idx2[head]]
		th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
		# interpolate
		th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].detach().cpu().numpy()

	attentions = attentions.reshape(nh, w_featmap, h_featmap)
	attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].detach().cpu().numpy()

	# save attentions heatmaps
	prefix = f'id{index}_' if index is not None else ''
	os.makedirs(args.output_dir, exist_ok=True)
	torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, f"img{frame_index}" + ".png"))
	img = Image.open(os.path.join(args.output_dir, f"img{frame_index}" + ".png"))

	attns = Image.new('RGB', (attentions.shape[2] * nh, attentions.shape[1]))
	for j in range(nh):
		#fname = os.path.join(args.output_dir, prefix + "attn-head" + str(j) + ".png")
		fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
		plt.imsave(fname=fname, arr=attentions[j], format='png')
		attns.paste(Image.open(fname), (j * attentions.shape[2], 0))

	return attentions, th_attn, img, attns

def show_attn_color(image, attentions, th_attn, index=None, head=[0,1,2,3,4,5]):
	M = image.max()
	m = image.min()
	span = 64
	image = ((image - m) / (M-m)) * span + (256 - span)
	image = image.mean(axis=2)
	image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
	
	for j in head:
		m = attentions[j]
		m *= th_attn[j]
		attentions[j] = m
	mask = np.stack([attentions[j] for j in head])
	
	blur = False
	contour = False
	alpha = 1
	figsize = tuple([i / 100 for i in args.image_size])
	fig = plt.figure(figsize=figsize, frameon=False, dpi=100)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	ax = plt.gca()

	if len(mask.shape) == 3:
		N = mask.shape[0]
	else:
		N = 1
		mask = mask[None, :, :]

	# AJ
	for i in range(N):
		mask[i] = mask[i] * ( mask[i] == np.amax(mask, axis=0))
	a = np.cumsum(mask, axis=0)
	for i in range(N):
		mask[i] = mask[i] * (mask[i] == a[i])
	
	colors = company_colors[:N]

	# Show area outside image boundaries.
	height, width = image.shape[:2]
	margin = 0
	ax.set_ylim(height + margin, -margin)
	ax.set_xlim(-margin, width + margin)
	ax.axis('off')
	masked_image = 0.1*image.astype(np.uint32).copy()
	for i in range(N):
		color = colors[i]
		_mask = mask[i]
		if blur:
			_mask = cv2.blur(_mask,(10,10))
		# Mask
		masked_image = apply_mask2(masked_image, _mask, color, alpha)
		# Mask Polygon
		# Pad to ensure proper polygons for masks that touch image edges.
		if contour:
			padded_mask = np.zeros(
				(_mask.shape[0] + 2, _mask.shape[1] + 2))#, dtype=np.uint8)
			padded_mask[1:-1, 1:-1] = _mask
			contours = find_contours(padded_mask, 0.5)
			for verts in contours:
				# Subtract the padding and flip (y, x) to (x, y)
				verts = np.fliplr(verts) - 1
				p = Polygon(verts, facecolor="none", edgecolor=color)
				ax.add_patch(p)
	ax.imshow(masked_image.astype(np.uint8), aspect='auto')
	ax.axis('image')
	#fname = os.path.join(output_dir, 'bnw-{:04d}'.format(imid))
	prefix = f'id{index}_' if index is not None else ''
	fname = os.path.join(args.output_dir, "attn_color.png")
	fig.savefig(fname)
	attn_color = Image.open(fname)

	return attn_color


def get_attention_map(model, img, get_mask=False):
	att_mat = model.get_last_selfattention(img, spatial=True)

	# Average the attention weights across all heads.
	att_mat = torch.mean(att_mat, dim=1).cpu()

	# To account for residual connections, we add an identity matrix to the
	# attention matrix and re-normalize the weights.
	residual_att = torch.eye(att_mat.size(1))
	aug_att_mat = att_mat + residual_att
	aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

	# Recursively multiply the weight matrices
	joint_attentions = torch.zeros(aug_att_mat.size())
	joint_attentions[0] = aug_att_mat[0]

	for n in range(1, aug_att_mat.size(0)):
		joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

	v = joint_attentions[-1]
	grid_size = int(np.sqrt(aug_att_mat.size(-1)))
	mask = v[0, :].reshape(grid_size, grid_size).detach().numpy()
	if get_mask:
		result = cv2.resize(mask / mask.max(), (img.shape[-2], img.shape[-1]))
	else:
		mask = cv2.resize(mask / mask.max(), (img.shape[-2], img.shape[-1]))[..., np.newaxis]
		result = (mask * (img[0, 0, ...].permute(1, 2, 0).cpu().numpy() * 255)).astype("uint8")

	return result

def get_topk_grid(model, img, topk=10):
	att_mat = model.get_last_selfattention(img, spatial=True)

	# Average the attention weights across all heads.
	att_mat = torch.mean(att_mat, dim=1).cpu()

	# To account for residual connections, we add an identity matrix to the
	# attention matrix and re-normalize the weights.
	residual_att = torch.eye(att_mat.size(1))
	aug_att_mat = att_mat + residual_att
	aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

	# Recursively multiply the weight matrices
	joint_attentions = torch.zeros(aug_att_mat.size())
	joint_attentions[0] = aug_att_mat[0]

	for n in range(1, aug_att_mat.size(0)):
		joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

	v = joint_attentions[-1]
	grid_size = int(np.sqrt(aug_att_mat.size(-1)))
	mask = v[0, :].reshape(grid_size, grid_size).detach().numpy()
	x, y = np.unravel_index(np.argsort(mask, axis=None), mask.shape)
	x = x[-topk:][::-1]
	y = y[-topk:][::-1]
	return x, y


def plot_attention_map(original_img, att_map):
	fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
	ax1.set_title('Original')
	ax2.set_title('Attention Map Last Layer')
	_ = ax1.imshow(original_img)
	_ = ax2.imshow(att_map)


if __name__ == '__main__':
	parser = argparse.ArgumentParser('Visualize Self-Attention maps')
	parser.add_argument('--arch', default='timesformer', type=str, choices=['timesformer', 'vivit'], help='Architecture.')
	parser.add_argument('--pretrained_weights', default='', type=str, help="""Path to pretrained 
		weights to evaluate. Set to `download` to automatically load the pretrained DINO from url.
		Otherwise the model is randomly initialized""")
	parser.add_argument('--output_dir', default='./attention_map', help='Path where to save visualizations.')
	parser.add_argument("--threshold", type=float, default=0.6, help="""We visualize masks
		obtained by thresholding the self-attention maps to keep xx% of the mass.""")
	parser.add_argument("--patch_size", type=int, default=16, help="""patch size.""")
	parser.add_argument("--image_size", default=(224, 224), type=int, nargs="+", help="Resize image.")
	parser.add_argument("--num_class", default=400, type=int, help="num classes")
	parser.add_argument('--num_frames', type=int, required=True, help='the mumber of frame sampling')
	parser.add_argument('--weights_from', type=str, default='imagenet', help='the pretrain params from [imagenet, kinetics]')
	args = parser.parse_args()

	#utils.fix_random_seeds(0)
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	
	# build model
	num_frames = args.num_frames
	frame_interval = 16
	num_class = args.num_class
	arch = args.arch # turn to vivit for initializing vivit model
	if arch == 'timesformer':
		pretrain_pth = args.pretrained_weights #'./timesformer_k400.pth'
		model = TimeSformer(num_frames=num_frames,
							img_size=args.image_size,
							patch_size=16,
							embed_dims=768,
							in_channels=3,
							attention_type='divided_space_time',
							return_cls_token=True)
	elif arch == 'vivit':
		pretrain_pth = args.pretrained_weights
		model = ViViT(
			pretrain_pth=pretrain_pth,
			img_size=args.image_size,
			num_frames=num_frames)
	else:
		raise TypeError(f'not supported arch type {arch}, chosen in (timesformer, vivit)')

	if args.weights_from == 'imagenet':
		weight_init.init_from_vit_pretrain_(model,
								pretrain_pth,
								'Conv3d',
								'fact_encoder',
								'repeat')
	elif args.weights_from == 'kinetics':
		weight_init.init_from_kinetics_pretrain_(model, pretrain_pth)
	else:
		raise TypeError(f'not supported weights_from {args.weights_from}, chosen in (imagenet, kinetics)')
	model.eval()
	model = model.to(device)
	print('load model finished')

	# build data
	video_path = './demo/YABnJL_bDzw.mp4'
	mean, std = (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)
	data_transform = T.Compose([
			T.Resize(scale_range=(-1, 256)),
			T.CenterCrop(args.image_size),
			T.ToTensor(),
			T.Normalize(mean, std)
			])
	temporal_sample = T.TemporalRandomCrop(num_frames*frame_interval)

	video_decoder = DecordInit()
	v_reader = video_decoder(video_path)
	total_frames = len(v_reader)
	start_frame_ind, end_frame_ind = temporal_sample(total_frames)
	if end_frame_ind-start_frame_ind < num_frames:
		raise ValueError(f'the total frames of the video {video_path} is less than {num_frames}')
	frame_indice = np.linspace(0, end_frame_ind-start_frame_ind-1, num_frames, dtype=int)
	video = v_reader.get_batch(frame_indice).asnumpy()
	del v_reader

	video = torch.from_numpy(video).permute(0,3,1,2) # Video transform: T C H W
	data_transform.randomize_parameters()
	video = data_transform(video)
	video = video.to(device)

	# extract the attention maps
	w_featmap = video.shape[-2] // args.patch_size
	h_featmap = video.shape[-1] // args.patch_size
	result = get_attention_map(model, video.unsqueeze(0).to(device))
	plot_attention_map(video[0].permute(1, 2, 0).cpu().numpy(), result)
	plt.show()
	print(get_topk_grid(model, video.unsqueeze(0).to(device)))
