#!/usr/bin/env python3

import numpy as np

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