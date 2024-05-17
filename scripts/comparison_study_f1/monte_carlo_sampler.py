import numpy as np
import casadi as ca

from DGSQP.types import VehicleState, VehicleActuation, Position, ParametricPose, OrientationEuler, BodyLinearVelocity, BodyAngularVelocity
from DGSQP.tracks.track_lib import get_track, load_tum_raceline

from globals import TRACK, VL, VW

track_obj = get_track(TRACK).get_track_segment([60, 80], 10)
H = track_obj.half_width
L = track_obj.track_length

path = f"traj_race_cl.csv"
raceline, s2t, raceline_mat = load_tum_raceline(path, TRACK, tenth_scale=True, segment=[60, 80])

pose_1 = ca.SX.sym('pose_1', 3) # (x, y, yaw)
pose_2 = ca.SX.sym('pose_2', 3)
r_1 = np.sqrt((VL/2)**2 + (VW/2)**2)
r_2 = np.sqrt((VL/2)**2 + (VW/2)**2)

collision_check_fn = ca.Function('collision_check', [pose_1, pose_2], [(r_1+r_2)**2 - ca.bilin(ca.DM.eye(2), pose_1[:2] - pose_2[:2],  pose_1[:2] - pose_2[:2])])

rng = np.random.default_rng(seed=0)

def get_sample():
    while True:
        car1_s = (L-10)*rng.random()
        _, _, _, _vx, _, _, _ep, _, _ey = np.array(raceline(s2t(car1_s))).squeeze()
        car1_ey = max(min(_ey + (2*rng.random() - 1), 0.9*float(track_obj.left_width(car1_s))), -0.9*float(track_obj.right_width(car1_s)))
        car1_ep = _ep
        car1_vx = _vx + (1.5*rng.random() - 0.75)
        car1_sim_state = VehicleState(t=0.0, 
                                p=ParametricPose(s=car1_s, x_tran=car1_ey, e_psi=car1_ep), 
                                v=BodyLinearVelocity(v_long=car1_vx))
        
        car2_s = car1_s + 3*rng.random()*VL
        _, _, _, _vx, _, _, _ep, _, _ey = np.array(raceline(s2t(car2_s))).squeeze()
        car2_ey = max(min(_ey + (2*rng.random() - 1), 0.9*float(track_obj.left_width(car2_s))), -0.9*float(track_obj.right_width(car2_s)))
        car2_ep = _ep
        car2_vx = _vx + (1.5*rng.random() - 0.75)
        car2_sim_state = VehicleState(t=0.0, 
                                p=ParametricPose(s=car2_s, x_tran=car2_ey, e_psi=car2_ep), 
                                v=BodyLinearVelocity(v_long=car2_vx))
        
        track_obj.local_to_global_typed(car1_sim_state)
        track_obj.local_to_global_typed(car2_sim_state)

        _p1 = np.array([car1_sim_state.x.x, car1_sim_state.x.y, car1_sim_state.e.psi])
        _p2 = np.array([car2_sim_state.x.x, car2_sim_state.x.y, car2_sim_state.e.psi])
        if collision_check_fn(_p1, _p2) >= 0:
            print('Sampled initial condition is in collision, resampling...')
        else:
            break

    return [car1_sim_state, car2_sim_state]