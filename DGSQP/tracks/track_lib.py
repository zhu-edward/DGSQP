#!/usr/bin/env python3

import numpy as np
import casadi as ca

import os
import pickle
import copy
import csv

from DGSQP.tracks.radius_arclength_track import RadiusArclengthTrack
from DGSQP.tracks.casadi_bspline_track import CasadiBSplineTrack

class StraightTrack(RadiusArclengthTrack):
    def __init__(self, length, width, slack, phase_out=False):
        if phase_out:
            cl_segs = np.array([[length, 0],
                                [10, 0]])
        else:
            cl_segs = np.array([length, 0]).reshape((1,-1))

        super().__init__(width, slack, cl_segs)
        self.phase_out = phase_out
        self.circuit = False
        self.initialize()

class CurveTrack(RadiusArclengthTrack):
    def __init__(self, enter_straight_length, 
                        curve_length, 
                        curve_swept_angle, 
                        exit_straight_length, 
                        width, 
                        slack,
                        phase_out=False,
                        ccw=True):
        if ccw:
            s = 1
        else:
            s = -1
        if phase_out:
            cl_segs = np.array([[enter_straight_length, 0],
                            [curve_length,          s*curve_length/curve_swept_angle],
                            [exit_straight_length,  0],
                                [10, 0]])
        else:
            cl_segs = np.array([[enter_straight_length, 0],
                                [curve_length, s * curve_length / curve_swept_angle],
                                [exit_straight_length, 0]])
        super().__init__(width, slack, cl_segs)
        self.phase_out = phase_out
        self.circuit = False
        self.initialize()

class ChicaneTrack(RadiusArclengthTrack):
    def __init__(self, enter_straight_length, 
                        curve1_length, 
                        curve1_swept_angle, 
                        mid_straight_length,
                        curve2_length,
                        curve2_swept_angle,
                        exit_straight_length,
                        width, 
                        slack,
                        phase_out=False,
                        mirror=False):
        if mirror:
            s1, s2 = 1, -1
        else:
            s1, s2 = -1, 1
        if phase_out:
            cl_segs = np.array([[enter_straight_length, 0],
                                [curve1_length, s1 * curve1_length / curve1_swept_angle],
                                [mid_straight_length, 0],
                                [curve2_length, s2 * curve2_length / curve2_swept_angle],
                                [exit_straight_length, 0],
                                [10, 0]])
        else:
            cl_segs = np.array([[enter_straight_length, 0],
                            [curve1_length,          s1*curve1_length/curve1_swept_angle],
                            [mid_straight_length,    0],
                            [curve2_length,          s2*curve2_length/curve2_swept_angle],
                            [exit_straight_length,   0]])

        super().__init__(width, slack, cl_segs)
        self.phase_out = phase_out
        self.circuit = False
        self.initialize()

def get_save_folder():
    return os.path.join(os.path.dirname(__file__), 'track_data')
    
def get_available_tracks():
    save_folder = get_save_folder()
    return os.listdir(save_folder)
    
def get_track(track_file):
    if not track_file.endswith('.npz'): track_file += '.npz'
    
    if track_file not in get_available_tracks():
        raise ValueError('Chosen Track is unavailable: %s\nlooking in:%s\n Available Tracks: %s'%(track_file, 
                        os.path.join(os.path.dirname(__file__), 'tracks', 'track_data'),
                        str(get_available_tracks())))

    save_folder = get_save_folder()
    load_file = os.path.join(save_folder,track_file)
    
    if track_file.endswith('.npz'):
        data = np.load(load_file, allow_pickle = True)
        if data['save_mode'] == 'radius_and_arc_length':
            track = RadiusArclengthTrack()
            track.initialize(data['track_width'], data['slack'], data['cl_segs'])
        elif data['save_mode'] == 'casadi_bspline':
            track = CasadiBSplineTrack(data['xy_waypoints'], data['left_width'], data['right_width'], 2.0, s_waypoints=data['s_waypoints'])
        else:
            raise NotImplementedError('Unknown track save mode: %s' % data['save_mode'])
    elif track_file.endswith('.pkl'):
        with open(load_file, 'rb') as f:
            track = pickle.load(f)
    else:
        raise ValueError(f'Unable to load track file {load_file}')
        
    return track   

def load_mpclab_raceline(file_path, track_name, time_scale=1.0):
    track = get_track(track_name)
    f = np.load(file_path)
    raceline_mat = np.vstack((f['x'], f['y'], f['psi'], f['v_long']/time_scale, f['v_tran']/time_scale, f['psidot']/time_scale, f['e_psi'], f['s'], f['e_y'])).T
    T = f['t']*time_scale

    raceline_mat2 = copy.copy(raceline_mat)
    raceline_mat2[:,7] += track.track_length
    T2 = copy.copy(T)
    T2 += T[-1]
    raceline_two_laps = np.vstack((raceline_mat, raceline_mat2[1:]))
    T_two_laps = np.append(T, T2[1:])
    t_sym = ca.MX.sym('t', 1)
    raceline_interp = []
    for i in range(raceline_mat.shape[1]):
        raceline_interp.append(ca.interpolant(f'x{i}', 'linear', [T_two_laps], raceline_two_laps[:,i]))
    raceline = ca.Function('raceline', [t_sym], [ri(t_sym) for ri in raceline_interp])
    s2t = ca.interpolant('s2t', 'linear', [raceline_two_laps[:,7]], T_two_laps)

    return raceline, s2t, raceline_mat

def load_tum_raceline(file_path, track_name, tenth_scale=False, time_scale=1.0, segment=None, resample_resolution=None):
    track = get_track(track_name)
    size_scale = 0.1 if tenth_scale else 1.0

    raceline_mat = []
    raceline_s = []
    with open(file_path, 'r') as f:
        _data = csv.reader(f, delimiter=';')
        for d in _data:
            if '#' in d[0]:
                continue
            _s, _x, _y, _psi, k, _v, a = [float(_d) for _d in d]
            x = _x*size_scale
            y = _y*size_scale
            v = _v*size_scale/time_scale
            psi = _psi + np.pi/2
            s, ey, epsi = track.global_to_local((x, y, psi))
            # pdb.set_trace()
            if len(raceline_mat) > 0:
                if s < raceline_mat[-1][7]:
                    s += track.track_length
            raceline_mat.append([x, y, psi, v, 0, 0, epsi, s, ey])
            raceline_s.append(_s*size_scale)
    raceline_mat = np.array(raceline_mat)
    T = [0.0]
    for k in range(len(raceline_s)-1):
        ds = raceline_s[k+1] - raceline_s[k]
        v = raceline_mat[k, 3]
        dt = ds/v
        T.append(T[-1]+dt)
    T = np.array(T)

    if not resample_resolution:
        resample_resolution = int(len(raceline_s)/raceline_s[-1])

    if segment:
        t_sym = ca.MX.sym('t', 1)

        _raceline_interp = []
        for i in range(raceline_mat.shape[1]):
            _raceline_interp.append(ca.interpolant(f'x{i}', 'linear', [T], raceline_mat[:,i]))
        _raceline = ca.Function('raceline', [t_sym], [ri(t_sym) for ri in _raceline_interp])
        _s2t = ca.interpolant('s2t', 'linear', [raceline_mat[:,7]], T)

        n = int((segment[1]-segment[0])*resample_resolution)
        _T = np.array(_s2t(np.linspace(segment[0], segment[1], n))).squeeze()
        T = _T - _T[0]
        raceline_mat = np.array(_raceline(_T)).squeeze().T
        raceline_mat[:,7] -= segment[0]
        raceline_interp = []
        for i in range(raceline_mat.shape[1]):
            raceline_interp.append(ca.interpolant(f'x{i}', 'linear', [T], raceline_mat[:,i]))
        raceline = ca.Function('raceline', [t_sym], [ri(t_sym) for ri in raceline_interp])
        s2t = ca.interpolant('s2t', 'linear', [raceline_mat[:,7]], T)
    else:
        raceline_mat2 = copy.copy(raceline_mat)
        raceline_mat2[:,7] += track.track_length
        T2 = copy.copy(T)
        T2 += T[-1]
        raceline_two_laps = np.vstack((raceline_mat, raceline_mat2[1:]))
        T_two_laps = np.append(T, T2[1:])
        t_sym = ca.MX.sym('t', 1)
        raceline_interp = []
        for i in range(raceline_mat.shape[1]):
            raceline_interp.append(ca.interpolant(f'x{i}', 'linear', [T_two_laps], raceline_two_laps[:,i]))
        raceline = ca.Function('raceline', [t_sym], [ri(t_sym) for ri in raceline_interp])
        s2t = ca.interpolant('s2t', 'linear', [raceline_two_laps[:,7]], T_two_laps)

    return raceline, s2t, raceline_mat