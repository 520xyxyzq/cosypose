# Get the error statistics of object pose predictions
# Ziqi Lu ziqilu@mit.edu
# Copyright 2023 The Ambitious Folks of the MRG

import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from collections import defaultdict
from tqdm import tqdm

from .calc_pose_error import pose_residual_error
from cosypose.config import BOP_TOOLKIT_DIR, LOCAL_DATA_DIR
from scipy.stats import norm

sys.path.append(str(BOP_TOOLKIT_DIR))
from bop_toolkit_lib import inout  # noqa


object_labels=[
    "002_master_chef_can_16k",
    "003_cracker_box_16k",
    "004_sugar_box_16k",
    "005_tomato_soup_can_16k",
    "006_mustard_bottle_16k",
    "007_tuna_fish_can_16k",
    "008_pudding_box_16k",
    "009_gelatin_box_16k",
    "010_potted_meat_can_16k",
    "011_banana_16k",
    "019_pitcher_base_16k",
    "021_bleach_cleanser_16k",
    "024_bowl_16k",
    "025_mug_16k",
    "035_power_drill_16k",
    "036_wood_block_16k",
    "037_scissors_16k",
    "040_large_marker_16k",
    "051_large_clamp_16k",
    "052_extra_large_clamp_16k",
    "061_foam_brick_16k"
]


def read_scene_gt(scene_gt):
    """
    Read scene_gt into dict
    """
    scene_gt_org = {}
    for im_id, gts in scene_gt.items():
        for gt in gts:
            scene_gt_org.setdefault(im_id, {}).setdefault(
                gt["obj_id"], []).append(gt)
    return scene_gt_org


def read_est(ests):
    """
    Read scene_gt into dict
    """
    ests_org = {}
    for est in ests:
        ests_org.setdefault(est['scene_id'], {}).setdefault(
        est['im_id'], {}).setdefault(est['obj_id'], []).append(est)
    assert len(ests_org.keys()) == 1, "Error: only support single scene!"
    return ests_org[list(ests_org.keys())[0]]


def eCDF(data, support=None):
    """
    Compute the empirical CDF from data
    @param data (N-array): Data
    @param support (M-array): Support over which to plot eCDF
    @return ecdf (Kx2 array): Empirical CDF
    """
    if support is None:
        min_, max_ = None, None
    else:
        min_, max_ = support[0], support[-1]
    data_sort = np.sort(data)
    ecdf = np.linspace(0, 1, len(data), endpoint=False)
    if min_ is not None and min_ < data_sort[0]:
        data_sort = np.insert(data_sort, 0, min_)
        ecdf = np.insert(ecdf, 0, 0.0)
    if max_ is not None and max_ > data_sort[-1]:
        data_sort = np.append(data_sort, max_)
        ecdf = np.append(ecdf, 1.0)
    return np.stack((data_sort, ecdf), axis=-1)


def errorPerSequence(real, pbr, gt, out):
    """
    YCBV per sequence error stats
    """
    real_est_files = sorted(glob.glob(f"{real}/*.csv"))
    pbr_est_files = sorted(glob.glob(f"{pbr}/*.csv"))
    gt_files = sorted(glob.glob(f"{gt}/*/"))
    for real_est_file, pbr_est_file, gt_file in tqdm(
        zip(real_est_files, pbr_est_files, gt_files)
    ):
        # Get YCB-V seq ID
        seq_id = int(os.path.basename(real_est_file)[:-4])
        # Read files
        real_est = inout.load_bop_results(real_est_file)
        pbr_est = inout.load_bop_results(pbr_est_file)
        scene_gt = inout.load_scene_gt(f"{gt_file}/scene_gt.json")
        # Compute residual errors in the tangential space of pose pred error
        real_errs = pose_residual_error(real_est, scene_gt)
        pbr_errs = pose_residual_error(pbr_est, scene_gt)
        obj_ids = set([e["obj_id"] for e in real_errs])
        for obj_id in obj_ids:
            # TODO: adapt this to work for duplicate (physical) objects
            # We only take first instance's pose error ie. e["errors"][0]
            real_err_obj = [
                e["errors"][0] for e in real_errs if e["obj_id"]==obj_id
            ]
            pbr_err_obj = [
                e["errors"][0] for e in pbr_errs if e["obj_id"]==obj_id
            ]
            # Make a 3x2 plot for error stats
            fig, (axes_xyz, axes_R123) = plt.subplots(nrows=2, ncols=3)
            support_xyz = np.arange(-10.01, 10.01, 0.01)
            support_R123 = np.arange(-np.pi-0.01, np.pi+0.01, 0.01)    
            real_data = np.array(real_err_obj)
            real_data[:, :3] *= 100 # [m] --> [cm]
            pbr_data = np.array(pbr_err_obj)
            pbr_data[:, :3] *= 100 # [m] --> [cm]
            # Fit a Gaussian distribution to data
            real_mean = np.mean(real_data, axis=0)
            real_cov = np.cov(real_data, rowvar=False, bias=True)
            pbr_mean = np.mean(pbr_data, axis=0)
            pbr_cov = np.cov(pbr_data, rowvar=False, bias=True)
            labels = [
                "X (cm)", "Y (cm)", "Z (cm)",
                "Rx (rad)", "Ry (rad)", "Rz (rad)"
            ]
            # Error stats for xyz
            for i in range(3):
                ecdf = eCDF(real_data[:, i], support_xyz)
                axes_xyz[i].plot(
                    ecdf[:, 0], ecdf[:, 1], color="k", linewidth=1
                )
                axes_xyz[i].plot(
                    support_xyz,
                    norm.cdf(support_xyz, real_mean[i], np.sqrt(real_cov[i, i])),
                    color='r', linestyle=":", linewidth=2
                )
                ecdf = eCDF(pbr_data[:, i], support_xyz)
                axes_xyz[i].plot(
                    ecdf[:, 0], ecdf[:, 1], color="b", linewidth=1
                )
                axes_xyz[i].plot(
                    support_xyz,
                    norm.cdf(support_xyz, pbr_mean[i], np.sqrt(pbr_cov[i, i])),
                    color='g', linestyle=":", linewidth=2
                )
                axes_xyz[i].set_xlim([support_xyz[0], support_xyz[-1]])
                axes_xyz[i].set_ylim([-0.05, 1.05])
                axes_xyz[i].set_xlabel(labels[i])
            # Error stats for lie algebra components R1 R2 R3
            for i in range(3):
                ecdf = eCDF(real_data[:, i+3], support_R123)
                axes_R123[i].plot(
                    ecdf[:, 0], ecdf[:, 1], color="k", linewidth=1
                )
                axes_R123[i].plot(
                    support_R123, norm.cdf(
                        support_R123, real_mean[i+3], np.sqrt(real_cov[i+3, i+3])
                    ), color='r', linestyle=":", linewidth=2
                )
                ecdf = eCDF(pbr_data[:, i+3], support_R123)
                axes_R123[i].plot(
                    ecdf[:, 0], ecdf[:, 1], color="b", linewidth=1
                )
                axes_R123[i].plot(
                    support_R123, norm.cdf(
                        support_R123, pbr_mean[i+3], np.sqrt(pbr_cov[i+3, i+3])
                    ), color='g', linestyle=":", linewidth=2
                )
                axes_R123[i].set_xticks(
                    [-np.pi, 0, np.pi], ["$-\pi$", "0", "$\pi$"]
                )
                axes_R123[i].set_xlim([support_R123[0], support_R123[-1]])
                axes_R123[i].set_ylim([-0.05, 1.05])
                axes_R123[i].set_xlabel(labels[i+3])
            fig.suptitle(f"YCBV {seq_id} Obj {obj_id} error stats")
            fig.legend([
                "Real Empirical CDF", "Real Fitted Normal CDF",
                "PBR Empirical CDF", "PBR Fitted Normal CDF"
            ])
            plt.tight_layout()
            plt.savefig(f"{out}/ycbv{seq_id}_obj{obj_id}.png", dpi=300)


def errorPerObject(real, pbr, gt, out):
    """
    YCBV per object error stats
    """
    real_est_files = sorted(glob.glob(f"{real}/*.csv"))
    pbr_est_files = sorted(glob.glob(f"{pbr}/*.csv"))
    gt_files = sorted(glob.glob(f"{gt}/*/"))
    real_err_obj_all = defaultdict(list)
    pbr_err_obj_all =  defaultdict(list)
    for real_est_file, pbr_est_file, gt_file in tqdm(
        zip(real_est_files, pbr_est_files, gt_files)
    ):
        # Read files
        real_est = inout.load_bop_results(real_est_file)
        pbr_est = inout.load_bop_results(pbr_est_file)
        scene_gt = inout.load_scene_gt(f"{gt_file}/scene_gt.json")
        # Compute residual errors in the tangential space of pose pred error
        real_errs = pose_residual_error(real_est, scene_gt)
        pbr_errs = pose_residual_error(pbr_est, scene_gt)
        obj_ids = set([e["obj_id"] for e in real_errs])
        for obj_id in obj_ids:
            # TODO: adapt this to work for duplicate (physical) objects
            # We only take first instance's pose error ie. e["errors"][0]
            real_err_obj = [
                e["errors"][0] for e in real_errs if e["obj_id"]==obj_id
            ]
            pbr_err_obj = [
                e["errors"][0] for e in pbr_errs if e["obj_id"]==obj_id
            ]
            real_err_obj_all[obj_id] += real_err_obj
            pbr_err_obj_all[obj_id] += pbr_err_obj

    # Plot per-object error stats figures (3 x #Obj)
    # NOTE: Change this to plot stats for different objects
    objs_to_plot=[2, 4, 10, 13, 16, 18] #real_err_obj_all.keys()
    fig_xyz, (axes_xyz) = plt.subplots(
        nrows=3, ncols=len(objs_to_plot), layout='constrained',
        figsize=(3.5 * len(objs_to_plot), 3.5 * 3), linewidth=5
    )
    fig_R123, (axes_R123) = plt.subplots(
        nrows=3, ncols=len(objs_to_plot), layout='constrained',
        figsize=(3.5 * len(objs_to_plot), 3.5 * 3), linewidth=5
    )
    support_xyz = np.arange(-10.01, 10.01, 0.01)
    support_R123 = np.arange(-np.pi-0.01, np.pi+0.01, 0.01) 
    labels = [
        "X (cm)", "Y (cm)", "Z (cm)",
        "Rx (rad)", "Ry (rad)", "Rz (rad)"
    ]
    for obj_ind, obj_id in enumerate(objs_to_plot):
        real_err_obj = real_err_obj_all[obj_id]
        pbr_err_obj = pbr_err_obj_all[obj_id]
        # Make a 3x2 plot for error stats
        real_data = np.array(real_err_obj)
        real_data[:, :3] *= 100 # [m] --> [cm]
        pbr_data = np.array(pbr_err_obj)
        pbr_data[:, :3] *= 100 # [m] --> [cm]
        # Fit a Gaussian distribution to data
        real_mean = np.mean(real_data, axis=0)
        real_cov = np.cov(real_data, rowvar=False, bias=True)
        pbr_mean = np.mean(pbr_data, axis=0)
        pbr_cov = np.cov(pbr_data, rowvar=False, bias=True)

        # Error stats for xyz
        for i in range(3):
            ecdf = eCDF(real_data[:, i], support_xyz)
            axes_xyz[i][obj_ind].plot(
                ecdf[:, 0], ecdf[:, 1], color="k", linewidth=2
            )
            axes_xyz[i][obj_ind].plot(
                support_xyz,
                norm.cdf(support_xyz, real_mean[i], np.sqrt(real_cov[i, i])),
                color='r', linestyle=":", linewidth=4
            )
            ecdf = eCDF(pbr_data[:, i], support_xyz)
            axes_xyz[i][obj_ind].plot(
                ecdf[:, 0], ecdf[:, 1], color="b", linewidth=2
            )
            axes_xyz[i][obj_ind].plot(
                support_xyz,
                norm.cdf(support_xyz, pbr_mean[i], np.sqrt(pbr_cov[i, i])),
                color='g', linestyle=":", linewidth=4
            )
            axes_xyz[i][obj_ind].set_xlim([support_xyz[0], support_xyz[-1]])
            axes_xyz[i][obj_ind].set_ylim([-0.05, 1.05])

            if not obj_ind == 0:
                axes_xyz[i][obj_ind].set_yticks([])
            if i == 2:
                axes_xyz[i][obj_ind].set_xlabel(
                    f"{labels[i]}\n{object_labels[obj_id-1]}", fontsize=15
                )
            else:
                axes_xyz[i][obj_ind].set_xlabel(
                    f"{labels[i]}", fontsize=15
                )
        # Error stats for lie algebra components R1 R2 R3
        for i in range(3):
            ecdf = eCDF(real_data[:, i+3], support_R123)
            axes_R123[i][obj_ind].plot(
                ecdf[:, 0], ecdf[:, 1], color="k", linewidth=2
            )
            axes_R123[i][obj_ind].plot(
                support_R123, norm.cdf(
                    support_R123, real_mean[i+3], np.sqrt(real_cov[i+3, i+3])
                ), color='r', linestyle=":", linewidth=4
            )
            ecdf = eCDF(pbr_data[:, i+3], support_R123)
            axes_R123[i][obj_ind].plot(
                ecdf[:, 0], ecdf[:, 1], color="b", linewidth=2
            )
            axes_R123[i][obj_ind].plot(
                support_R123, norm.cdf(
                    support_R123, pbr_mean[i+3], np.sqrt(pbr_cov[i+3, i+3])
                ), color='g', linestyle=":", linewidth=4
            )
            # axes_R123[i][obj_id-1].set_xticks(
            #     [-np.pi, 0, np.pi], ["$-\pi$", "0", "$\pi$"]
            # )
            axes_R123[i][obj_ind].set_xlim(
                [support_R123[0], support_R123[-1]]
            )
            axes_R123[i][obj_ind].set_ylim([-0.05, 1.05])            
            if not obj_ind == 0:
                axes_R123[i][obj_ind].set_yticks([])
            if i == 2:
                axes_R123[i][obj_ind].set_xlabel(
                    f"{labels[i+3]}\n{object_labels[obj_id-1]}", fontsize=15
                )
            else:
                axes_R123[i][obj_ind].set_xlabel(
                    f"{labels[i+3]}", fontsize=15
                )
        fig_xyz.suptitle(
            f"YCBV translation prediction error stats", fontsize=30
        )
        fig_xyz.legend([
            "Real Empirical CDF", "Real Fitted Normal CDF",
            "PBR Empirical CDF", "PBR Fitted Normal CDF"
        ], ncol=2, fontsize=15, framealpha=0.5)
        fig_xyz.tight_layout()
        fig_xyz.savefig(f"{out}/trans_error.png", dpi=100)
        fig_R123.suptitle(
            f"YCBV rotation prediction error stats", fontsize=30
        )
        fig_R123.legend([
            "Real Empirical CDF", "Real Fitted Normal CDF",
            "PBR Empirical CDF", "PBR Fitted Normal CDF"
        ], ncol=2, fontsize=15, framealpha=0.5)
        fig_R123.tight_layout()
        fig_R123.savefig(f"{out}/orien_error.png", dpi=100)


def confidencePerSequence(est, gt, out):
    """
    YCBV per sequence confidence stats
    """
    est_files = sorted(glob.glob(f"{est}/*.csv"))
    gt_files = sorted(glob.glob(f"{gt}/*/"))
    for est_file, gt_file in tqdm(zip(est_files, gt_files)):
        estimates = inout.load_bop_results(est_file)
        scene_gt = inout.load_scene_gt(f"{gt_file}/scene_gt.json")
        est_dict = read_est(estimates)
        scene_gt_dict = read_scene_gt(scene_gt)
        seq_id = int(os.path.basename(est_file)[:-4])

        obj_scores = defaultdict(list)
        for im_id, im_targets in est_dict.items():
            for obj_id, target in im_targets.items():
                obj_scores[obj_id].append(target[0]["score"])

        num_imgs = len(scene_gt_dict)
        obj_exist = scene_gt_dict[1].keys()
        # NOTE: Change the detection confidence thresholds
        thresholds = [0.30, 0.60, 0.90]
        percent_preds = np.zeros((len(object_labels), len(thresholds)))
        for obj_id, obj_score in sorted(obj_scores.items()):
            percent_preds[obj_id-1, :] = [
                len([sc for sc in obj_score if sc > th]) / num_imgs 
                for th in thresholds
            ]
        for ii in range(len(thresholds)):
            plt.bar([ol[4:-4] for ol in object_labels], percent_preds[:, ii])
            plt.xticks(rotation="vertical")
            for ind, xticklabel in enumerate(plt.gca().get_xticklabels()):
                if ind + 1 in obj_exist:
                    xticklabel.set_color("green")
                else:
                    xticklabel.set_color("red")
        plt.legend(
            [str(th) for th in thresholds], title="Confidence Thresholds",
            ncol=len(thresholds)
        )
        plt.title(
            f"YCB-V {seq_id} "+
            f"object detection w/ different confidence thresholds"
        )
        plt.ylabel("Frames with detections (%)")
        plt.ylim([0.0, 1.0])
        plt.tight_layout()
        plt.savefig(f"{out}/ycbv{seq_id}_detect_percent.png", dpi=300)
        plt.clf()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real", "-r", type=str,
        help="Path to csv files with pose preds from real data trained model",
        default='/home/ziqi/Desktop/test/real/'
    )
    parser.add_argument(
        "--pbr", "-p", type=str,
        help="Path to csv files with pose preds from real data trained model",
        default='/home/ziqi/Desktop/test/pbr/'
    )
    parser.add_argument(
        "--gt", "-g", help="Path to scene_gt.json",
        default='/media/ziqi/Extreme SSD/data/bop_datasets/ycbv/train_real/'
    )
    parser.add_argument(
        "--out", "-o", default="/home/ziqi/Desktop/test", 
        help="Path to save the stats fig"
    )
    args = parser.parse_args()

    # Per-seq error stats
    # errorPerSequence(args.real, args.pbr, args.gt, args.out)
    # Per-object error stats
    # errorPerObject(args.real, args.pbr, args.gt, args.out)
    # Prediction confidence stats
    confidencePerSequence(args.pbr, args.gt, args.out)
    # Prediction accuracy vs confidence





