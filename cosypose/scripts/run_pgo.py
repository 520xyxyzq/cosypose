# Solve the pose graph optimization
# Ziqi Lu ziqilu@mit.edu

import argparse
import sys

import gtsam
import imageio
import json
import numpy as np
import pandas as pd
import pypose as pp
import torch
from collections import Counter, defaultdict
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.stats.distributions import chi2

import cosypose.utils.tensor_collection as tc
from cosypose.config import BOP_TOOLKIT_DIR, LOCAL_DATA_DIR
# from cosypose.datasets.bop_object_datasets import BOPObjectDataset
# from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.visualization.multiview import make_scene_renderings

sys.path.append(str(BOP_TOOLKIT_DIR))
from bop_toolkit_lib import inout  # noqa

# For GTSAM symbols
O = gtsam.symbol_shorthand.O
C = gtsam.symbol_shorthand.C


def tc_to_csv(predictions, csv_path):
    preds = []
    for n in range(len(predictions)):
        TCO_n = predictions.poses[n]
        t = TCO_n[:3, -1] * 1e3  # m -> mm conversion
        R = TCO_n[:3, :3]
        row = predictions.infos.iloc[n]
        obj_id = int(row.label.split('_')[-1])
        score = row.score
        time = -1.0
        pred = dict(scene_id=row.scene_id,
                    im_id=row.view_id,
                    obj_id=obj_id,
                    score=score,
                    t=t, R=R, time=time)
        preds.append(pred)
    inout.save_bop_results(csv_path, preds)


def read_csv_candidates(csv_path):
    df = pd.read_csv(csv_path)
    infos = df.loc[:, ['im_id', 'scene_id', 'score', 'obj_id']]
    infos['obj_id'] = infos['obj_id'].apply(lambda x: f'obj_{x:06d}')
    infos = infos.rename(dict(im_id='view_id', obj_id='label'), axis=1)
    R = np.stack(
        df['R'].apply(lambda x: list(map(float, x.split(' '))))
    ).reshape(-1, 3, 3)
    t = np.stack(
        df['t'].apply(lambda x: list(map(float, x.split(' '))))
    ).reshape(-1, 3) * 1e-3
    R = torch.tensor(R, dtype=torch.float)
    t = torch.tensor(t, dtype=torch.float)
    TCO = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(len(R), 1, 1)
    TCO[:, :3, :3] = R
    TCO[:, :3, -1] = t
    candidates = tc.PandasTensorCollection(poses=TCO, infos=infos)
    return candidates


def read_cameras(json_path, view_ids, read_pose=False):
    '''
    Read camera info from json file(s)
    @param json_path: [str] Path to the BOP-format scene_camera.json file
    @param view_ids: [list] Indices for the camera views to read
    @param read_pose: [bool] Whether to read the camera poses in json file
    '''
    cameras = json.loads(Path(json_path).read_text())
    all_K = []
    all_cam_pose = []
    for view_id in view_ids:
        cam_info = cameras[str(view_id)]
        K = np.array(cam_info['cam_K']).reshape(3, 3)
        all_K.append(K)
        if read_pose:
            rot = torch.tensor(cam_info['cam_R_w2c'])
            trans = torch.tensor(cam_info['cam_t_w2c'])
            pose = torch.eye(4).to(torch.float)
            pose[:3, :3] = rot.reshape(3, 3).to(torch.float)
            pose[:3, 3] = trans.reshape(3).to(torch.float) * 1e-3
            pose = pose.inverse().unsqueeze(0)  # w2c --> c2w
            all_cam_pose.append(pose)

    K = torch.as_tensor(np.stack(all_K))

    if read_pose:
        cam_poses = torch.cat(all_cam_pose, dim=0)
        cameras = tc.PandasTensorCollection(
            K=K, infos=pd.DataFrame(dict(view_id=view_ids)),
            TWC=cam_poses
        )
    else:
        cameras = tc.PandasTensorCollection(
            K=K, infos=pd.DataFrame(dict(view_id=view_ids))
        )
    return cameras


def save_scene_json(objects, cameras, results_scene_path):
    list_cameras = []
    list_objects = []

    for n in range(len(objects)):
        obj = objects.infos.loc[n, ['score', 'label', 'n_cand']].to_dict()
        obj = {k: np.asarray(v).item() for k, v in obj.items()}
        obj['TWO'] = objects.TWO[n].cpu().numpy().tolist()
        list_objects.append(obj)

    for n in range(len(cameras)):
        cam = cameras.infos.loc[n, ['view_id']].to_dict()
        cam['TWC'] = cameras.TWC[n].cpu().numpy().tolist()
        cam['K'] = cameras.K[n].cpu().numpy().tolist()
        list_cameras.append(cam)

    scene = dict(objects=list_objects, cameras=list_cameras)
    results_scene_path.write_text(json.dumps(scene))
    return


class GTSAMPGO:
    """
    Use GTSAM to solve the object-based bundle adjustment problem
    """

    def __init__(self, candidates, cameras):
        """
        Read camera, object and pose prediction information
        @param candidates (PandasTensorCollection): Pose prediction info
        @param cameras (PandasTensorCollection): Camera info
        """
        assert Counter(cameras.infos['view_id'].values).keys() == \
            Counter(candidates.infos['view_id'].values).keys(), \
            "Error: view_ids mismatch in scene_camera.json and candidates.csv"
        # Read camera information
        self.K = cameras.K
        self.cam_infos = cameras.infos
        self.view_ids = self.cam_infos['view_id']
        # View ID (e.g. 10, 20, 40, ...) to index (e.g. 0, 1, 2, ...)
        self.view_id_to_index = dict(
            zip(self.view_ids, np.arange(len(self.view_ids)))
        )
        self.TWC = cameras.TWC
        self.n_views = len(self.cam_infos)
        
        # Read object information
        # Data association based on semantic measurements
        # TODO: relax this assumption
        self.obj_labels = np.unique(candidates.infos['label'].values)
        self.n_objects = len(self.obj_labels)
        self.obj_indices = np.arange(self.n_objects, dtype=int)
        self.obj_ids = [int(label[4:]) for label in self.obj_labels]
        self.obj_id_to_index = dict(zip(self.obj_ids, self.obj_indices))
        self.obj_label_to_index = dict(zip(self.obj_labels, self.obj_indices))
        # self.obj_points = self.mesh_db.select(self.obj_labels).points
        # self.n_points = self.obj_points.shape[1]
        # Object confidence scores as sum of confidence scores (for viz only)
        self.obj_scores = candidates.infos.groupby('label').sum()
        self.obj_scores = self.obj_scores['score'].values

        # Read object pose prediction information
        self.cand_TCO = candidates.poses
        self.cand_view_ids = candidates.infos['view_id'].values
        self.cand_labels = candidates.infos['label'].values
        self.cand_view_indices = [
            self.view_id_to_index[view_id] for view_id in self.cand_view_ids
        ]
        self.cand_obj_indices = [
            self.obj_label_to_index[label] for label in self.cand_labels
        ]
        self.n_candidates = len(self.cand_TCO)
        # HashMap (Obj ID, View ID) : Obj-to-cam pose preds
        self._co_TCO_map_ = {
            (c, o): TCO for (c, o, TCO) in zip(
                self.cand_view_indices[::-1], # Iterate in reversed order
                self.cand_obj_indices[::-1], # to only remember the higher
                self.cand_TCO.flip(dims=(0,)) # confidence predictions
            )
        }
        # Which object is observed in each camera view
        self._detected_ = defaultdict(list)
        for cid, oid in zip(self.cand_view_indices, self.cand_obj_indices):
            self._detected_[cid].append(oid)
        
        # self.residuals_ids = self.make_residuals_ids()

        # Build pose graph
        self._fg_ = gtsam.NonlinearFactorGraph()
        self._init_ = gtsam.Values()
        # Initialize camera and object pose variables
        self.initVariables()
        # Add factors to PGO
        self.addFactors(kernel="Cauchy")

    def initVariables(self):
        """
        Initialize object landmark variable and camera pose variables in PGO
        """
        # Init camera pose variables
        for cid, cam_pose in enumerate(self.TWC):
            cam_pose = gtsam.Pose3(cam_pose.cpu())
            self._init_.insert(C(cid), cam_pose)        
        # Init object pose variables
        TO_init = defaultdict(list)
        for (cid, oid), TCO in self._co_TCO_map_.items():
            obj_pose = gtsam.Pose3((self.TWC[cid] @ TCO).cpu())
            TO_init[oid].append(obj_pose)
        # Initialize object pose as average pose predictions
        for oid, obj_poses in TO_init.items():
            avg_obj_pose = self.avgPoses(obj_poses)
            self._init_.insert(O(oid), avg_obj_pose)

    def addFactors(self, kernel="Cauchy"):
        """
        Add factors to pose graph
        @param kernel (str): Robust kernel used for object pose pred factors
        TODO: add support for L2 cost function
        """
        # Add camera pose prior and odometry factors
        for cid, cam_pose in enumerate(self.TWC):
            cam_pose_gtsam = gtsam.Pose3(cam_pose.cpu())
            if cid == 0:
                self._fg_.add(
                    gtsam.PriorFactorPose3(
                        C(cid), gtsam.Pose3(cam_pose_gtsam),
                        gtsam.noiseModel.Isotropic.Sigma(6, 1e-10)
                    )
                )
            else:
                rel_pose = prev_cam_pose.inverse().compose(cam_pose_gtsam)
                self._fg_.add(
                    gtsam.BetweenFactorPose3(
                        C(cid - 1), C(cid), rel_pose,
                        gtsam.noiseModel.Isotropic.Sigma(6, 0.01)
                    )
                )
            prev_cam_pose = cam_pose_gtsam
        # Add pose prediction factors
        # TODO: find a way to specify the noise model
        noise_model = gtsam.noiseModel.Isotropic.Sigma(6, 0.05)
        robust = self.robustFunction(kernel)
        robust_nm = gtsam.noiseModel.Robust(robust, noise_model)
        for (cid, oid), pred in self._co_TCO_map_.items():
            self._fg_.add(
                gtsam.BetweenFactorPose3(
                    C(cid), O(oid), gtsam.Pose3(pred.cpu()), robust_nm
                )
            )

    def robustFunction(self, kernel="Cauchy"):
        '''
        Define robust noise model based on noise model
        @param kernel (str): Robust kernel name
        @return robust (gtsam.noiseModel.mEstimator): Robust cost function
        '''
        if kernel == "Cauchy":
            robust = gtsam.noiseModel.mEstimator.Cauchy(1.0)
        elif kernel == "Huber":
            robust = gtsam.noiseModel.mEstimator.Huber(1.345)
        elif kernel == "GemanMcClure":
            robust = gtsam.noiseModel.mEstimator.GemanMcClure(1.0)
        elif kernel == "Tukey":
            robust = gtsam.noiseModel.mEstimator.Tukey(4.6851)
        elif kernel == "Welsch":
            robust = gtsam.noiseModel.mEstimator.Welsch(2.9846)
        else:
            raise ValueError(f"Unknown kernel type: {kernel}")
        return robust

    def avgPoses(self, poses):
        """
        Average a list of poses
        @param poses (list of gtsam.Pose3): Poses to average
        @return avg_pose (gtsam.Pose3): Average pose
        """
        assert type(poses) is list and len(poses) > 0, \
            "Error: Pose list is empty"
        t = np.mean(np.array([pose.translation() for pose in poses]), axis=0)
        quats = np.array([
            pose.rotation().toQuaternion().coeffs() for pose in poses
        ])
        # NOTE: GTSAM toQuaternion.coeffs quat order xyzw
        # NOTE: scipy Rotation quat order xyzw
        # NOTE: So no need to convert quat order here
        # Rotation Averaging
        quat = R.from_quat(quats).mean().as_quat()
        # NOTE: but GTSAM Quaternion order wxyz, so we do need to convert here
        # scipy -> GTSAM Rotation quat order xyzw->wxyz
        quat = np.hstack((quat[-1], quat[:3]))
        return gtsam.Pose3(gtsam.Rot3.Quaternion(*quat), gtsam.Point3(t))

    def solve(self, optimizer="LM", verbose=False):
        """
        Solve robust pose graph optimization
        @param optimizer: [str] NLS optimizer for pose graph optimization
        @param verbose: [bool] Print optimization stats?
        @return result: [gtsam.Values] Optimization result
        """
        if optimizer == "GN":
            params = gtsam.GaussNewtonParams()
            if verbose:
                params.setVerbosity("ERROR")
            optim = gtsam.GaussNewtonOptimizer(self._fg_, self._init_, params)
        elif optimizer == "LM":
            params = gtsam.LevenbergMarquardtParams()
            if verbose:
                params.setVerbosity("ERROR")
            optim = gtsam.LevenbergMarquardtOptimizer(
                self._fg_, self._init_, params
            )
        elif optimizer == "GNCLM":
            params = gtsam.LevenbergMarquardtParams()
            params = gtsam.GncLMParams(params)
            params.setVerbosityGNC(params.Verbosity.SILENT)
            optim = gtsam.GncLMOptimizer(self._fg_, self._init_, params)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer}")

        self._result_ = optim.optimize()
        TWO_opt, TWC_opt = self.valuesToPose(self._result_)
        return TWO_opt, TWC_opt

    def checkObjExist(
        self, values, kernel="Cauchy", conf_mat=[[0.9, 0.1], [0.1, 0.9]]
    ):
        """
        Decision making for whether object landmarks exist
        @param values (gtsam.Values): GTSAM Values
        @param kernel (str): Robust kernel name
        @parm conf_mat (list or array): Confusion matrix 
        @return exist (list): Whether object i exist
        """
        nll0s, nll1s = defaultdict(float), defaultdict(float)
        chi2inv = chi2.ppf(0.95, df=6)
        robust = self.robustFunction(kernel)
        # Accumulate NLLs for pose prediction factors
        for fac_ind in range(self._fg_.size()):
            fac = self._fg_.at(fac_ind)
            key_vec = fac.keys()
            if len(key_vec) == 2 and \
                gtsam.Symbol(key_vec[-1]).chr() == ord("o"):
                obj_ind = gtsam.Symbol(key_vec[-1]).index()
                nll1s[obj_ind] += fac.error(values)
                nll0s[obj_ind] += robust.loss(np.sqrt(chi2inv))
        # Accumulate NLLs for landmark existence factors
        # TODO: Check visibility
        for cid in range(self.n_views):
            for oid in self.obj_indices:
                if oid in self._detected_[cid]:
                    nll0s[oid] += -np.log(conf_mat[0][1])
                    nll1s[oid] += -np.log(conf_mat[1][1])
                else:
                    nll0s[oid] += -np.log(conf_mat[0][0])
                    nll1s[oid] += -np.log(conf_mat[1][0])
        decisions, exist_prob = defaultdict(bool), defaultdict(float)
        for (_, nll0), (obj_ind, nll1) in zip(nll0s.items(), nll1s.items()):
            prob0 = np.exp(-nll0) / (np.exp(-nll0) + np.exp(-nll1))
            prob1 = 1 - prob0
            exist_prob[obj_ind] = prob1
            decisions[obj_ind] = prob0 < prob1
        return decisions, exist_prob

    def valuesToPose(self, values):
        """
        Extract object and camera poses from GTSAM Values
        @param values: [gtsam.Values] Pose values
        @return TWO_opt: [Ox4x4 tensor] Object-to-world poses
        @return TWC_opt: [Cx4x4 tensor] Object-to-world poses
        """
        # Read object and camera pose values
        # No need to sort the variables by their keys
        # GTSAM already sorted them
        TWO, TWC = [], []
        for key in values.keys():
            if gtsam.Symbol(key).chr() == ord('o'):
                two = values.atPose3(key)
                two = torch.tensor(two.matrix()).unsqueeze(0)
                TWO.append(two)
            if gtsam.Symbol(key).chr() == ord('c'):
                twc = values.atPose3(key)
                twc = torch.tensor(twc.matrix()).unsqueeze(0)
                TWC.append(twc)
        TWO_torch = torch.cat(TWO, dim=0)
        TWC_torch = torch.cat(TWC, dim=0)
        return TWO_torch, TWC_torch

    def valuesToPandasTensorCollection(self, values, exist=None):
        """
        Convert values to PandasTensorCollections for visualization
        @param values (gtsam.Values): GTSAM Values
        @param exist (dict {int: bool} or None): Which object exists
        @return objects (PandasTensorCollection): Object information
        @return cameras (PandasTensorCollection): Camera information
        @return reproj (PandasTensorCollection): Obj-to-cam poses info
        """
        TWO, TWC = self.valuesToPose(values)
        if exist is None:
            exist = {oi: True for oi in self.obj_indices}
        obj_indices = [
            gtsam.Symbol(key).index() for key in values.keys() 
            if gtsam.Symbol(key).chr() == ord('o') and 
            exist[gtsam.Symbol(key).index()]
        ]
        TWO = TWO[obj_indices]
        obj_labels = [self.obj_labels[obj_index] for obj_index in obj_indices]
        obj_scores = [self.obj_scores[obj_index] for obj_index in obj_indices]
        obj_infos = pd.DataFrame({
            'obj_id': obj_indices, 'label': obj_labels, 'score': obj_scores,
            'view_group': np.zeros(len(obj_indices), int)
        })
        cam_infos = pd.DataFrame({
            'view_id': self.view_ids, 'batch_im_id': np.arange(self.n_views),
            'scene_id': np.zeros(len(self.view_ids), int)
        })
        objects = tc.PandasTensorCollection(
            poses=TWO, TWO=TWO, infos=obj_infos
        )
        cameras = tc.PandasTensorCollection(TWC=TWC, infos=cam_infos)
        # Obtain obj-to-cam poses
        TCO_data = []
        for o in range(len(objects)):
            for v in range(len(cameras)):
                obj = objects[[o]]
                cam = cameras[[v]]
                infos = dict(
                    scene_id=cam.infos['scene_id'].values,
                    view_id=cam.infos['view_id'].values,
                    score=obj.infos['score'].values + 1.0,
                    view_group=obj.infos['view_group'].values,
                    label=obj.infos['label'].values,
                    batch_im_id=cam.infos['batch_im_id'].values,
                    obj_id=obj.infos['obj_id'].values,
                    from_ba=[True],
                )
                data_ = tc.PandasTensorCollection(
                    infos=pd.DataFrame(infos),
                    poses=cam.TWC.inverse() @ obj.TWO
                )
                TCO_data.append(data_)
        reproj = tc.concatenate(TCO_data)
        return objects, cameras, reproj

    def plot(self, result):
        """
        Plot estimation results
        @param values (gtsam.Values): PGO results
        """
        obj_poses, cam_poses = self.valuesToPose(result)
        obj_poses, cam_poses = obj_poses.numpy(), cam_poses.numpy()
        axes = plt.figure().add_subplot(projection='3d')
        # Plot optimized camera poses
        axes.plot3D(
            cam_poses[:, 0, -1], cam_poses[:, 1, -1], cam_poses[:, 2, -1],
            "b-", linewidth=2, label="Traj. after PGO"
        )
        axes.scatter3D(
            obj_poses[:, 0, -1], obj_poses[:, 1, -1], obj_poses[:, 2, -1],
            "r.", linewidth=2, label="Object positions"
        )
        # axes.view_init(azim=-90, elev=-45)
        axes.legend()
        plt.show()

    def make_residuals_ids(self):
        cand_ids, obj_ids, view_ids, point_ids, xy_ids = [], [], [], [], []
        for cand_id in range(self.n_candidates):
            for point_id in range(self.n_points):
                for xy_id in range(2):
                    cand_ids.append(cand_id)
                    obj_ids.append(self.cand_obj_ids[cand_id])
                    view_ids.append(self.cand_view_ids[cand_id])
                    point_ids.append(point_id)
                    xy_ids.append(xy_id)
        residuals_ids = dict(
            cand_id=cand_ids,
            obj_id=obj_ids,
            view_id=view_ids,
            point_id=point_ids,
            xy_id=xy_ids,
        )
        return residuals_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scenario', default='', type=str,
        help="scenario dir name, must match a folder in local_data/scenarios"
    )
    parser.add_argument(
        '--sv_score_th', default=0.3, type=float,
        help="Score to filter single-view predictions"
    )
    parser.add_argument(
        '--n_symmetries_rot', default=64, type=int,
        help="Number of discretized symmetries to use for continuous symmetry"
    )
    parser.add_argument(
        '--ba_n_iter', default=10, type=int,
        help="Maximum number of LM iterations in stage 3"
    )
    parser.add_argument(
        '--nms_th', default=0.04, type=float,
        help='Threshold (meter) for NMS 3D'
    )
    parser.add_argument('--no_visualization', action='store_true')
    args = parser.parse_args()

    scenario_dir = LOCAL_DATA_DIR / 'custom_scenarios' / args.scenario

    candidates = read_csv_candidates(scenario_dir / 'candidates.csv')
    candidates = candidates.float().cuda()
    candidates.infos['group_id'] = 0
    scene_ids = np.unique(candidates.infos['scene_id'])
    assert len(scene_ids) == 1, \
        'Please only provide 6D pose estimations from the same scene.'
    scene_id = scene_ids.item()
    view_ids = np.unique(candidates.infos['view_id'])
    n_views = len(view_ids)
    print(f'Loaded {len(candidates)} candidates in {n_views} views.')

    cameras = read_cameras(
        scenario_dir / 'scene_camera.json', view_ids, read_pose=True
    ).float().cuda()
    cameras.infos['scene_id'] = scene_id
    cameras.infos['batch_im_id'] = np.arange(len(view_ids))
    print(f'Loaded cameras intrinsics.')

    # object_ds = BOPObjectDataset(scenario_dir / 'models')
    # mesh_db = MeshDataBase.from_object_ds(object_ds)
    # logger.info(f'Loaded {len(object_ds)} 3D object models.')

    print('Running stage 2 and 3 of CosyPose...')

    # Keep pose predictions with confidence above threshold
    keep = np.where(candidates.infos['score'] >= args.sv_score_th)[0]
    candidates = candidates[keep]

    gtsam_pgo = GTSAMPGO(candidates, cameras)
    gtsam_pgo.solve(verbose=True)
    # gtsam_pgo.plot(gtsam_pgo._result_)
    # gtsam_pgo.plot(gtsam_pgo._init_)
    decision, prob = gtsam_pgo.checkObjExist(gtsam_pgo._result_)

    objects_, cameras_, reproj_ = gtsam_pgo.valuesToPandasTensorCollection(
        gtsam_pgo._result_, decision
    )
    tc_to_csv(reproj_, f"{scenario_dir}/results/reproj.csv")
    print(objects_)

    fps = 25
    duration = 10
    n_images = fps * duration
    # n_images = 1  # Uncomment this if you just want to look at one image
    images = make_scene_renderings(
        objects_, cameras_, urdf_ds_name='ycbv',
        distance=1.3, object_scale=2.0, use_nms3d=True,
        show_cameras=False, camera_color=(0, 0, 0, 1),
        theta=np.pi/4, resolution=(640, 480), object_id_ref=4,
        # TODO: make this work for other datasets, e.g. tless
        colormap_rgb=defaultdict(lambda: [1, 1, 1, 1]),
        angles=np.linspace(0, 2*np.pi, n_images)
    )
    imageio.mimsave(f"{scenario_dir}/results/res.gif", images, fps=fps)

    # Calculate and print pose prediction errors
    from cosypose.scripts.calc_pose_error import err_calc_simple

    estimates = inout.load_bop_results(f"{scenario_dir}/results/reproj.csv")
    scene_gt = inout.load_scene_gt(f"{scenario_dir}/scene_gt.json")
    errs = err_calc_simple(estimates, scene_gt)
    obj_ids = set([e["obj_id"] for e in errs])
    errs_object = {}
    print("Median pose errors:")
    for obj_id in obj_ids:
        # TODO: adapt this to work for duplicate (physical) objects
        # We only take first instance's pose error ie. e["errors"][0]
        err_obj = [e["errors"][0] for e in errs if e["obj_id"]==obj_id]
        errs_object[obj_id] = err_obj
        err_obj_med = np.median(np.array(err_obj), axis=0)
        print(
            f"Obj {obj_id}: " +
            f"R_err (deg): {err_obj_med[0]}, t_err (cm): {err_obj_med[1]}"
        )
    # Save pose errors
    errs_sorted = sorted(errs, key=lambda x: x["obj_id"])
    with open(f"{scenario_dir}/results/error.json", "w+") as fp:
        json.dump(errs_sorted, fp, indent=4, sort_keys=False)