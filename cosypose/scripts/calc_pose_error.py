
# BOP dataset evaluation requires specific file formats e.g. filename format
# We provide simplified functions for easier pose error calculation
import sys
from cosypose.config import BOP_TOOLKIT_DIR

sys.path.append(str(BOP_TOOLKIT_DIR))
from bop_toolkit_lib import pose_error


def err_calc_simple(ests, scene_gt):
    """
    Simplified error computation script adapted from BOP dataset
    @param ests: [dict] Loaded bop results from .csv file
    @param scene_gt: [dict] Loaded object ground truth poses from .json file
    @return errs: [list of dicts] Calculated pose errors
    """
    # Organize the estimates into recursive dicts
    ests_org = {}
    for est in ests:
        ests_org.setdefault(est['scene_id'], {}).setdefault(
        est['im_id'], {}).setdefault(est['obj_id'], []).append(est)
    assert len(ests_org.keys()) == 1, "Error: only support single scene!"
    scene_id = list(ests_org)[0]

    # Organize the gt object poses into recursive dicts
    scene_gt_org = {}
    for im_id, gts in scene_gt.items():
        for gt in gts:
            scene_gt_org.setdefault(im_id, {}).setdefault(gt["obj_id"], []).append(gt)

    # Compute error per object per image
    scene_errs = []
    for im_ind, (im_id, im_targets) in enumerate(scene_gt_org.items()):
        for obj_id, target in im_targets.items():
            n_top_curr = sum([gt['obj_id'] == obj_id for gt in target])
            # Get object pose estimates
            try:
                obj_ests = ests_org[scene_id][im_id][obj_id]
            except KeyError:
                obj_ests = []

            # Sort the estimates by score (in descending order).
            obj_ests_sorted = sorted(
                enumerate(obj_ests), key=lambda x: x[1]['score'], reverse=True
            )
            # Get top n_top_curr estimates
            obj_ests_sorted = obj_ests_sorted[slice(0, n_top_curr)]

            for est_id, est in obj_ests_sorted:
                # Estimated pose.
                R_e = est['R']
                t_e = est['t']

                errs = {}
                for gt_id, gt in enumerate(target):
                    if gt['obj_id'] != obj_id:
                        continue
                    # Ground-truth pose.
                    R_g = gt['cam_R_m2c']
                    t_g = gt['cam_t_m2c']
                    # compute trans. and rot. error
                    e = [pose_error.re(R_e, R_g), pose_error.te(t_e, t_g)]
                    errs[gt_id] = e
                # Scene errors
                scene_errs.append({
                    'im_id': im_id,
                    'obj_id': obj_id,
                    'est_id': est_id,
                    'score': est['score'],
                    'errors': errs
                })
    return scene_errs