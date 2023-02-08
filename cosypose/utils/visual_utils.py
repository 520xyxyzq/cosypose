# Visualize the 2D detections and 6D pose predictions
# Author: Ziqi Lu ziqilu@mit.edu
# Copyright 2023 The Ambitious Folks of the MRG

import matplotlib.colors as mcolors
import numpy as np
import torch

from PIL import ImageColor
from torchvision.utils import draw_bounding_boxes
from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer


def drawDetections(
    image, detections, intrinsics=None, score_th=0.3, width=5, fontsize=100,
    alpha=0.7, render_cad=True
):
    """
    Draw 2D bbox and/or mask and/or 6D pose detections on a SINGLE image
    @param image (CxWxH): Image to draw on
    @param detections (PandasTensorCollection): Detections or predictions
    @param intrinsics (1x3x3): Camera intrinsics matrix
    @param score_th (float): Detection score threshold, below which don't draw
    @param width (float): Bbox line width
    @param fontsize (int): The bbox label font size (points)
    @param alpha (float): Transparency of the mask
    @param render_cad (bool): Render object as CAD if True else ren pure color
    """
    labels = detections.infos["label"]
    scores = detections.infos["score"]
    # Only Viz bboxes beyond score threshold
    keep = np.where(scores > score_th)[0]
    labels = labels[keep]
    # Squeeze the batch dimension if any
    if len(image.shape) == 4:
        assert image.shape[0] == 1, "Error: only support single img"
        image = image.squeeze(0)
    # Convert image to 0~255 if it's not 
    if image.dtype in [torch.float, torch.float32]:
        image  = (image * 255).byte()
    # Generate object colors based on object label IDs
    colors = list(mcolors.TABLEAU_COLORS.values()) * 10
    colors_obj, colors_obj_rgb = [], []
    for l in labels:
        # Extract color based on label ID if any, otherwise black
        color = colors[int(l[-3:])] if l[-3:].isnumeric() else '#000000'
        color_rgb = ImageColor.getrgb(color)
        color_rgb = torch.tensor(color_rgb).byte()
        colors_obj.append(color)
        colors_obj_rgb.append(color_rgb)
    if hasattr(detections, "boxes_rend"):
        bboxes = detections.boxes_rend
        bboxes = bboxes[keep, ...]
        image = draw_bounding_boxes(
            image.cpu(), bboxes, labels,
            colors=colors_obj, width=width, font_size=fontsize
        )
    if hasattr(detections, "masks"):
        masks = detections.masks
        masks = masks[keep, ...]
        img_to_draw = image.detach().clone()
        for mask, color in zip(masks, colors_obj_rgb):
            img_to_draw[:, mask] = color[:, None]
        image = image * (1 - alpha) + img_to_draw * alpha
        image = image.byte()
    if hasattr(detections, "poses"):
        poses = detections.poses.to(image.device)
        assert intrinsics is not None, "Error: need intrinsics to render pose"
        # To make the occlusion relation right, sort poses and labels by depth
        depths = poses[:, 2, -1]
        order = torch.argsort(-depths)
        poses = poses[order, ...]
        labels = [list(labels)[o] for o in order]
        colors_obj_rgb = [colors_obj_rgb[o] for o in order]
        # Initialize a renderer
        # TODO: make object_set a param
        renderer = BulletBatchRenderer(
            object_set="ycbv", n_workers=8,
            gpu_renderer="cuda" in image.device.type
        )
        # Visualize the object poses
        img_size = tuple(image.shape[-2:])
        img_to_draw = image.detach().clone()
        for obj_label, pose, color in zip(labels, poses, colors_obj_rgb):
            rgb_ren = renderer.render(
                [{'name': obj_label}], pose.unsqueeze(0), intrinsics, img_size
            ) # 1x3xHxW
            rgb_ren = rgb_ren[0, ...].mul(255).byte() # 3xHxW
            mask_ren = rgb_ren[0, ...] > 0.0 # HxW
            if render_cad: # render object CAD model to represent the pose
                img_to_draw[:, mask_ren] = rgb_ren[:, mask_ren]
            else: # render pure color to represent the pose
                img_to_draw[:, mask_ren] = color[:, None].to(image.device)
            image = image * (1 - alpha) + img_to_draw * alpha
            image = image.byte()
    # Convert image to HxWxC format
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    return image