"""Per-clip quality numbers for a retargeted motion (lower is better for all of them).

foot_slide_m : how far feet move horizontally while touching the floor (should be ~0)
root_path_m  : total distance the pelvis travels; compare with net_m (start-to-end)
yaw_range_deg: how much the robot's heading changes over the clip
jitter_p95   : 95th percentile of joint acceleration (rad/s^2); a shaky clip scores high
vel_max      : fastest joint speed in the clip (rad/s)
"""
import numpy as np, mujoco
def quality(model_path, qpos, fps):
    m=mujoco.MjModel.from_xml_path(model_path); d=mujoco.MjData(m)
    ids=[m.body("left_ankle_roll_link").id, m.body("right_ankle_roll_link").id]
    F=[]
    for q in qpos:
        d.qpos[:]=q; mujoco.mj_kinematics(m,d); F.append([d.xpos[i].copy() for i in ids])
    F=np.array(F); zmin=F[:,:,2].min()
    slide=0.0
    for k in range(2):
        c=F[:,k,2]<zmin+0.015; both=c[1:]&c[:-1]
        slide+=np.linalg.norm(np.diff(F[:,k,:2],axis=0),axis=1)[both].sum()
    root=qpos[:,:2]; path=np.linalg.norm(np.diff(root,axis=0),axis=1).sum()
    w,x,y,z=qpos[:,3:7].T; yaw=np.degrees(np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))))
    v=np.diff(qpos[:,7:],axis=0)*fps; acc=np.diff(v,axis=0)*fps
    return dict(root_path_m=path, net_m=float(np.linalg.norm(root[-1]-root[0])), foot_slide_m=slide,
                yaw_range_deg=float(np.ptp(yaw)), jitter_p95=float(np.percentile(np.abs(acc),95)), vel_max=float(np.abs(v).max()))


def report(model_path, qpos, fps):
    q = quality(model_path, qpos, fps)
    print("[quality] " + " | ".join(f"{k} {v:.2f}" for k, v in q.items()))
    return q
