# Cleaner Cosypose pose inference code
# Borrowed from Jingnan Shi

import argparse
import glob
import imageio
import numpy as np
import torch
import json
import yaml
import pandas as pd
from copy import deepcopy
from enum import IntEnum
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from itertools import groupby

from cosypose.datasets.datasets_cfg import make_scene_dataset, make_object_dataset

# Pose estimator
from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.training.pose_models_cfg import create_model_refiner, create_model_coarse
from cosypose.training.pose_models_cfg import check_update_config as check_update_config_pose
from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
from cosypose.integrated.pose_predictor import CoarseRefinePosePredictor
# from cosypose.integrated.multiview_predictor import MultiviewScenePredictor
# from cosypose.datasets.wrappers.multiview_wrapper import MultiViewWrapper

# Detection
from cosypose.training.detector_models_cfg import create_model_detector
from cosypose.training.detector_models_cfg import check_update_config as check_update_config_detector
from cosypose.integrated.detector import Detector
from cosypose.utils.visual_utils import drawDetections
# from cosypose.evaluation.pred_runner.bop_predictions import BopPredictionRunner

# from cosypose.utils.distributed import get_tmp_dir, get_rank
# from cosypose.utils.distributed import init_distributed_mode
from cosypose.utils.tensor_collection import concatenate

from cosypose.config import EXP_DIR, RESULTS_DIR
from .run_custom_scenario import tc_to_csv

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


modelIds = {
    "ycbv": {
        "real":
        [
            'detector-bop-ycbv-synt+real--292971',
            'coarse-bop-ycbv-synt+real--822463',
            'refiner-bop-ycbv-synt+real--631598'
        ],
        "pbr":
        [
            'detector-bop-ycbv-pbr--970850',
            'coarse-bop-ycbv-pbr--724183',
            'refiner-bop-ycbv-pbr--604090'
        ],
    },
    "tless": {
        "real":[
            'detector-bop-tless-synt+real--452847',
            'coarse-bop-tless-synt+real--160982',
            'refiner-bop-tless-synt+real--881314'
        ],
        "pbr":[
            'detector-bop-tless-pbr--873074',
            'coarse-bop-tless-pbr--506801',
            'refiner-bop-tless-pbr--233420'
        ]
    }
}


def load_detector(run_id):
    run_dir = EXP_DIR / run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.Loader)
    cfg = check_update_config_detector(cfg)
    label_to_category_id = cfg.label_to_category_id
    model = create_model_detector(cfg, len(label_to_category_id))
    ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
    ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model = model.cuda().eval()
    model.cfg = cfg
    model.config = cfg
    model = Detector(model)
    return model


def load_pose_models(coarse_run_id, refiner_run_id=None, n_workers=8):
    run_dir = EXP_DIR / coarse_run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.Loader)
    cfg = check_update_config_pose(cfg)
    # object_ds = BOPObjectDataset(BOP_DS_DIR / 'tless/models_cad')
    object_ds = make_object_dataset(cfg.object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    renderer = BulletBatchRenderer(
        object_set=cfg.urdf_ds_name,
        n_workers=n_workers
    )
    mesh_db_batched = mesh_db.batched().cuda()

    def load_model(run_id):
        if run_id is None:
            return
        run_dir = EXP_DIR / run_id
        cfg = yaml.load(
            (run_dir / 'config.yaml').read_text(), Loader=yaml.Loader
        )
        cfg = check_update_config_pose(cfg)
        if cfg.train_refiner:
            model = create_model_refiner(
                cfg, renderer=renderer, mesh_db=mesh_db_batched
            )
        else:
            model = create_model_coarse(
                cfg, renderer=renderer, mesh_db=mesh_db_batched
            )
        ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
        ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt)
        model = model.cuda().eval()
        model.cfg = cfg
        model.config = cfg
        return model

    coarse_model = load_model(coarse_run_id)
    refiner_model = load_model(refiner_run_id)
    model = CoarseRefinePosePredictor(
        coarse_model=coarse_model,
        refiner_model=refiner_model
    )
    return model, mesh_db


def getModel(detector_id, coarse_id, refiner_id):
    """
    Load detector, pose estimation and refinment models;
    @param detector_id (str): ID for the 2D object detector
    @param coarse_id (str): ID for 6D object pose estimator
    @param refiner_id (str): ID for 6D object pose refinement model 
    """
    # load models
    detector = load_detector(detector_id)
    pose_predictor, mesh_db = load_pose_models(
        coarse_run_id=coarse_id, refiner_run_id=refiner_id, n_workers=4
    )
    return detector, pose_predictor


def binary_mask_to_rle(binary_mask):
    """
    Converts a binary mask to COCOs run-length encoding (RLE) format.
    Instead of outputting a mask image,
    you give a list of start pixels and how many pixels after each of those
    starts are included in the mask.

    :param binary_mask: a 2D binary numpy array where '1's represent the object
    :return: Mask in RLE format
    """
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(
            groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def saveCocoDetections(dets, json_file):
    """
    Save 2D detections to COCO-format file
    @param dets (PandasTensorCollect): 2D Bboxes + 2D Segmentations + labels
    @param json_file (str): Path to save the detection json file
    """
    assert len(dets) > 0, "Error: no detections to save"
    dets_lst = []
    for n in range(len(dets)):
        det = dets.infos.iloc[n]
        bbox = dets.bboxes[n].tolist()
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        mask = dets.masks[n].cpu().numpy()
        mask_rle = binary_mask_to_rle(mask)
        det_dict = dict(
            bbox=bbox, category_id=int(det.label.split('_')[-1]),
            image_id=int(det.batch_im_id), scene_id=0, score=det.score,
            segmentation=mask_rle, time=1.0
        )
        dets_lst.append(det_dict)
    with open(json_file, "w+") as fp:
        fp.write("[\n")
        fp.writelines('  ' + json.dumps(dic) + ',\n' for dic in dets_lst[:-1])
        fp.write('  ' + json.dumps(dets_lst[-1]) + "\n")
        fp.write("]")


def inference(
    detector, pose_predictor, image, camera_k, detection_th=0.3,
    one_instance_per_class=False
):
    """
    CosyPose pose inference code
    @param detector: CosyPose detector
    @param pose_predictor: CosyPose CoarseRefine pose predictor
    @param image ((Bx)3xHxW): Input RGB images
    @param camera_k ((1x)3x3): Camera intrinsics
    @param detection_th (float): 2D object detection confidence threshold
    @param one_instance_per_class (bool): Only use higher-confidence detection
    """
    images = image
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    # [1,3,3]
    cameras_k = camera_k.to(images.device)
    if len(cameras_k.shape) == 2:
        cameras_k = cameras_k.unsqueeze(0)
    # 2D object detection
    box_detections = detector.get_detections(
        images=images, one_instance_per_class=one_instance_per_class,
        detection_th=detection_th, output_masks=False, mask_th=0.9
    )
    # 6D pose esitimition
    if len(box_detections) == 0:
        return None
    # all_preds is preds at different refinement steps
    final_preds, all_preds = pose_predictor.get_predictions(
        images, cameras_k, detections=box_detections,
        n_coarse_iterations=1, n_refiner_iterations=4
    )

def predict(pose_predictor, image, camera_k, TCO_init,
            n_coarse_iterations=1, n_refiner_iterations=1):
    """
    Predict object poses based on initial guesses
    NOTE: only support single images for now
    @param pose_predictor (CoarseRefine): Coarse + Refiner pose predictor
    @param image ((Bx)3xHxW): Input RGB image
    @param camera_k ((1x)3x3): Camera intrinsics
    @param TCO_init (dict or PandasTensorCollection): Initial pose estimates
    @param n_coarse_iterations (int): Number of coarse pred iterations
    @param n_refiner_iterations (int): Number of refiner pred iterations
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(0) # [1,3,3]
    camera_k = camera_k.to(image.device)
    if len(camera_k.shape) == 2:
        camera_k = camera_k.unsqueeze(0)
    if type(TCO_init) is dict:
        labels = list(TCO_init.keys())
        poses  = torch.stack(list(TCO_init.values()), dim=0)
        TCO_init = PandasTensorCollection(
            poses = poses.to(image.device),
            infos=pd.DataFrame(dict(batch_im_id=[0]*len(labels), label=labels))
        )
    elif type(TCO_init) is PandasTensorCollection:
        TCO_init.poses = TCO_init.poses.to(image.device)
    else:
        raise ValueError(f"{type(TCO_init)} is not supported!")
    coarse_preds = pose_predictor.batched_model_predictions(
        pose_predictor.coarse_model, image, camera_k, TCO_init, 
        n_iterations=n_coarse_iterations
    )
    data_TCO = coarse_preds[f'iteration={n_coarse_iterations}']
    refiner_preds = pose_predictor.batched_model_predictions(
        pose_predictor.refiner_model, image, camera_k, data_TCO,
        n_iterations=n_refiner_iterations
    )
    data_TCO = refiner_preds[f'iteration={n_refiner_iterations}']
    return data_TCO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path", "-i", type=str, help="Path to the target image folder"
    )
    parser.add_argument(
        "--K", "-K", help="Camera intrinsics: fx, fy, cx, cy, s", nargs=5,
        type=float, default=[1066.778, 1067.487, 312.9869, 241.3109, 0]
    )
    parser.add_argument(
        "--data", "-d", type=str, default="ycbv",
        help="Dataset that the target objects belongs to"
    )
    parser.add_argument(
        "--train", "-t", type=str, default="real",
        help="Trained data type (real or pbr)"
    )
    parser.add_argument(
        "--plot", "-p", help="Viz the detections and poses? (run slower)",
        action="store_true"
    )
    parser.add_argument(
        "--out", "-o", default="/home/ziqi/Desktop/candidates.csv",
        help="File path to save the pose prediction results"
    )
    args = parser.parse_args()
    model_ids = modelIds[args.data][args.train]
    detector,pose_predictor = getModel(*model_ids)
    print("start...........................................")

    # Read images
    img_names = sorted(
        glob.glob(f"{args.img_path}/*.png") +
        glob.glob(f"{args.img_path}/*.jpg")
    )
    # Read intrinsics
    K = torch.eye(3)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1] = args.K
    K = K.unsqueeze(0)
    # Run inference
    # TODO: Change this to set image id
    view_ids = np.arange(1, len(img_names) + 1)
    preds, detections, imgs_to_viz = [], [], []
    for img_ind, img_name in enumerate(tqdm(img_names)):
        img = Image.open(img_name)
        img = np.array(img)
        img = torch.from_numpy(img).cuda().float().unsqueeze_(0)
        img = img.permute(0, 3, 1, 2) / 255
        # predict
        detection = detector.get_detections(
            images=img, one_instance_per_class=False,
            detection_th=0, output_masks=True
        )
        pred = inference(detector, pose_predictor, img, K)
        if pred is None:
            if args.plot:
                img_no_pred = np.array(Image.open(img_name))
                imgs_to_viz.append(img_no_pred)
            continue
        pred.infos["batch_im_id"] = [img_ind] * len(pred)
        pred.infos["scene_id"] = [0] * len(pred)
        pred.infos["view_id"] = [view_ids[img_ind]] * len(pred)
        detection.infos["batch_im_id"] = [img_ind] * len(detection)
        detection.infos["scene_id"] = [0] * len(detection)
        detection.infos["view_id"] = [view_ids[img_ind]] * len(detection)
        if args.plot:
            img_ren = drawDetections(img, pred.cuda(), K)
            imgs_to_viz.append(img_ren)
        preds.append(pred)
        detections.append(detection)
    preds = concatenate(preds)
    detections = concatenate(detections)
    if args.plot:
        imageio.mimwrite(
            f"{args.out}_video.mp4", imgs_to_viz, fps=2, quality=8
        )
    tc_to_csv(preds, args.out)
    saveCocoDetections(detections, args.out.replace(".csv", ".json"))
    # Refine
    obj_preds = read_csv_candidates(args.out)
    obj_preds.infos['batch_im_id'] = 0
    preds = []
    for img_ind, img_name in enumerate(tqdm(img_names)):
        img = Image.open(img_name)
        img = np.array(img)
        img = torch.from_numpy(img).cuda().float().unsqueeze_(0)
        img = img.permute(0, 3, 1, 2) / 255
        # Get the predictions on image indexed img_ind 
        keep = np.where(obj_preds.infos['view_id'] == img_ind + 1)[0]
        obj_preds_frame = obj_preds[keep]
        # predict
        pred = predict(pose_predictor, img, K, obj_preds_frame, 4, 4)
        pred.infos["batch_im_id"] = [img_ind] * len(pred)
        pred.infos["scene_id"] = [0] * len(pred)
        pred.infos["view_id"] = [view_ids[img_ind]] * len(pred)
        preds.append(pred)
    preds = concatenate(preds)
    tc_to_csv(preds, args.out)


if __name__ == '__main__':
    main()
