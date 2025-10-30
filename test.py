#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as osp
import argparse
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms

from model import BiSeNet

PART_COLORS = np.array([
    [0,   0,   0],    # 0 background (unused)
    [255, 0,   0],    # 1 skin
    [255, 85,  0],    # 2 l_brow
    [255, 170, 0],    # 3 r_brow
    [255, 0,   85],   # 4 l_eye
    [255, 0,   170],  # 5 r_eye
    [0,   255, 0],    # 6 eye_g
    [85,  255, 0],    # 7 l_ear
    [170, 255, 0],    # 8 r_ear
    [0,   255, 85],   # 9 ear_r
    [0,   255, 170],  # 10 nose
    [0,   0,   255],  # 11 mouth
    [85,  0,   255],  # 12 upper_lip
    [170, 0,   255],  # 13 lower_lip
    [0,   85,  255],  # 14 neck
    [0,   170, 255],  # 15 neck_l
    [255, 255, 0],    # 16 cloth
    [255, 255, 85],   # 17 hair
    [255, 255, 170],  # 18 hat
], dtype=np.uint8)

NOSE_ID = 10
MOUTH_IDS = [11, 12, 13]   # mouth / upper_lip / lower_lip

def choose_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def vis_parsing_on_image(im_rgb, parsing, save_path):
    h, w = parsing.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    uniq = np.unique(parsing)
    for cls in uniq:
        if cls < len(PART_COLORS):
            color_map[parsing == cls] = PART_COLORS[cls]
    overlay = cv2.addWeighted(cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR), 0.4, color_map, 0.6, 0)
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def has_parts(parsing, min_pixels=200, rel_thresh=0.0005):
    """返回 (has_mouth, has_nose)"""
    h, w = parsing.shape
    area = h * w
    thr = max(min_pixels, int(area * rel_thresh))  # 绝对像素兜底 + 相对阈值
    nose = (parsing == NOSE_ID).sum() > thr
    mouth = sum((parsing == i).sum() for i in MOUTH_IDS) > thr
    return mouth, nose

def evaluate(dspth, respth, ckpt_name):
    os.makedirs(respth, exist_ok=True)
    device = choose_device()
    print(f"[device] {device}")

    n_classes = 19
    net = BiSeNet(n_classes=n_classes).to(device).eval()

    # NOTE: 关键修复：map_location=device
    ckpt_path = ckpt_name if osp.isabs(ckpt_name) else osp.join('res/cp', ckpt_name)
    assert osp.exists(ckpt_path), f"ckpt not found: {ckpt_path}"
    net.load_state_dict(torch.load(ckpt_path, map_location=device))

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    names = [n for n in os.listdir(dspth) if osp.splitext(n.lower())[1] in exts]
    names.sort()
    for name in names:
        pil = Image.open(osp.join(dspth, name)).convert('RGB')
        im_rgb = np.array(pil.resize((512, 512), Image.BILINEAR))
        x = to_tensor(Image.fromarray(im_rgb)).unsqueeze(0).to(device)

        with torch.no_grad():
            out = net(x)[0]
            parsing = out.squeeze(0).detach().cpu().numpy().argmax(0).astype(np.uint8)

        has_mouth, has_nose = has_parts(parsing)
        print(f"{name}: has_mouth={has_mouth}, has_nose={has_nose}, uniq={np.unique(parsing)}")

        vis_parsing_on_image(im_rgb, parsing, save_path=osp.join(respth, name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dspth', required=True, help='dir of input images')
    parser.add_argument('--respth', default='./res/test_res', help='dir to save visualizations')
    parser.add_argument('--ckpt', default='79999_iter.pth', help='checkpoint file (abs path or res/cp/xxx.pth)')
    args = parser.parse_args()

    evaluate(dspth=args.dspth, respth=args.respth, ckpt_name=args.ckpt)
