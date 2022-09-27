#!/usr/bin/env python3

import numpy as np

import os

from DGSQP.tracks.radius_arclength_track import RadiusArclengthTrack

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
    
    npzfile = np.load(load_file, allow_pickle = True)
    if npzfile['save_mode'] == 'radius_and_arc_length':
        track = RadiusArclengthTrack()
        track.initialize(npzfile['track_width'], npzfile['slack'], npzfile['cl_segs'])
    else:
        raise NotImplementedError('Unknown track save mode: %s'%npzfile['save_mode'])
        
    return track   