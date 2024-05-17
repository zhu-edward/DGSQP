#!/usr/bin/env python3

import numpy as np
import scipy
from scipy.integrate import quad

import casadi as ca

from DGSQP.tracks.base_track import BaseTrack
        
class CasadiBSplineTrack(BaseTrack):
    def __init__(self, xy_waypoints, left_width, right_width, slack, 
                 s_waypoints=np.array([]), 
                 t_waypoints=np.array([]),
                 code_gen=False):
        # xy waypoints is an array of shape (N, 2)
        self.xy_waypoints = xy_waypoints
        self.left_width_points = left_width
        self.right_width_points = right_width
        self.track_width = np.mean(left_width + right_width)
        self.half_width = self.track_width/2
        self.slack = slack
        self.circuit = False
        self.code_gen = code_gen

        if np.allclose(self.xy_waypoints[0], self.xy_waypoints[-1]):
            self.circuit = True

        spline_options = dict(degree=[3])

        if len(s_waypoints):
            # If arclength values are provided, we can fit the spline directly
            self.s_waypoints = s_waypoints
        else: 
            if not len(t_waypoints):
                # If neither are provided, assume uniform spacing
                t_waypoints = np.linspace(0, 1, self.xy_waypoints.shape[0])
            # If timestamps are provided
            _x_spline = ca.interpolant('x_spline', 'bspline', [t_waypoints], self.xy_waypoints[:,0], spline_options)
            _y_spline = ca.interpolant('y_spline', 'bspline', [t_waypoints], self.xy_waypoints[:,1], spline_options)
            # Compute derivatives of scaled track
            _t = ca.MX.sym('t', 1)
            _dxdt = ca.Function('dxdt', [_t], [ca.jacobian(_x_spline(_t), _t)])
            _dydt = ca.Function('dydt', [_t], [ca.jacobian(_y_spline(_t), _t)])

            # Integrate over time to get distance
            _v = ca.Function('v', [_t], [ca.sqrt(_dxdt(_t)**2 + _dydt(_t)**2)])
            _D = lambda t: float(_v(t))

            self.s_waypoints = [0]
            for i in range(len(t_waypoints)-1):
                ds, e = quad(_D, t_waypoints[i], t_waypoints[i+1])
                self.s_waypoints.append(self.s_waypoints[-1]+ds)
            self.s_waypoints = np.array(self.s_waypoints)

        self.x = ca.interpolant('x_spline', 'bspline', [self.s_waypoints], self.xy_waypoints[:,0], spline_options)
        self.y = ca.interpolant('y_spline', 'bspline', [self.s_waypoints], self.xy_waypoints[:,1], spline_options)
        
        self.track_length = self.s_waypoints[-1] - self.s_waypoints[0]

        s_sym = ca.SX.sym('s', 1)
        s_bar = ca.fmod(ca.fmod(s_sym, self.track_length) + self.track_length, self.track_length)
        self.left_width = ca.Function('left_width', [s_sym], [ca.pw_lin(s_bar, self.s_waypoints, left_width)])
        self.right_width = ca.Function('right_width', [s_sym], [ca.pw_lin(s_bar, self.s_waypoints, right_width)])

        options = dict(jit=self.code_gen, jit_name='spline_track', compiler='shell', jit_options=dict(compiler='gcc', flags=['-O3'], verbose=False))
        s_sym = ca.MX.sym('s', 1)
        self.dx = ca.Function('dx', [s_sym], [ca.jacobian(self.x(s_sym), s_sym)], options)
        self.dy = ca.Function('dy', [s_sym], [ca.jacobian(self.y(s_sym), s_sym)], options)
        self.ddx = ca.Function('ddx', [s_sym], [ca.jacobian(self.dx(s_sym), s_sym)], options)
        self.ddy = ca.Function('ddy', [s_sym], [ca.jacobian(self.dy(s_sym), s_sym)], options)

        # Set up optimization problem for global to local conversion
        xy_sym = ca.MX.sym('xy', 2)
        xy = ca.vertcat(self.x(s_sym), self.y(s_sym))
        objective = ca.bilin(np.eye(2), xy_sym - xy, xy_sym - xy)
        prob = {'x': s_sym, 'f': objective, 'p': xy_sym}
        ipopt_opts = dict(print_level=0)
        solver_opts = dict(error_on_fail=False, 
                        ipopt=ipopt_opts, 
                        verbose=False, 
                        print_time=False, 
                        verbose_init=False)
        self.global_to_local_solver = ca.nlpsol('g2l', 'ipopt', prob, solver_opts)

        xi, yi, xo, yo = [], [], [], []
        for s in self.s_waypoints:
            _xi, _yi, _ = self.local_to_global((s, float(self.left_width(s)), 0))
            xi.append(_xi)
            yi.append(_yi)
            _xo, _yo, _ = self.local_to_global((s, -float(self.right_width(s)), 0))
            xo.append(_xo)
            yo.append(_yo)
        self.xi = ca.interpolant('xi_s', 'bspline', [self.s_waypoints], xi, spline_options)
        self.yi = ca.interpolant('yi_s', 'bspline', [self.s_waypoints], yi, spline_options)
        self.xo = ca.interpolant('xo_s', 'bspline', [self.s_waypoints], xo, spline_options)
        self.yo = ca.interpolant('yo_s', 'bspline', [self.s_waypoints], yo, spline_options)

    def get_closest_waypoint_index(self, xy):
        dist = []
        for _xy in self.xy_waypoints:
            dist.append(np.linalg.norm(xy - _xy))
        idx = np.argmin(dist)
        return idx
    
    def project_to_centerline(self, xy):
        _i = self.get_closest_waypoint_index(xy)
        sol = self.global_to_local_solver(x0=self.s_waypoints[_i], lbx=0, ubx=self.track_length, p=xy)
        success = self.global_to_local_solver.stats()['success']
        if not success:
            raise(ValueError(self.global_to_local_solver.stats()['return_status']))
        return float(sol['x'])
    
    def get_curvature(self, s):
        dx = float(self.dx(s))
        dy = float(self.dy(s))
        ddx = float(self.ddx(s))
        ddy = float(self.ddy(s))
        c = (dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
        return c
    
    def get_curvature_casadi_fn(self):
        sym_s = ca.MX.sym('s', 1)
        # Makes sure s is within [0, track_length]
        sym_s_bar = ca.fmod(ca.fmod(sym_s, self.track_length) + self.track_length, self.track_length)
        
        dx = self.dx(sym_s_bar)
        dy = self.dy(sym_s_bar)
        ddx = self.ddx(sym_s_bar)
        ddy = self.ddy(sym_s_bar)
        curvature = (dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)

        options = dict(jit=self.code_gen, jit_name='curvature', compiler='shell', jit_options=dict(compiler='gcc', flags=['-O3'], verbose=False))
        return ca.Function('track_curvature', [sym_s], [curvature], options)

    def get_tangent_angle_casadi_fn(self):        
        sym_s = ca.MX.sym('s', 1)
        # Makes sure s is within [0, track_length]
        sym_s_bar = ca.fmod(ca.fmod(sym_s, self.track_length) + self.track_length, self.track_length)

        dxy = ca.vertcat(self.dx(sym_s_bar), self.dy(sym_s_bar))
        n = dxy / ca.norm_2(dxy)
        track_angle = ca.atan2(n[1], n[0])

        options = dict(jit=self.code_gen, jit_name='tangent', compiler='shell', jit_options=dict(compiler='gcc', flags=['-O3'], verbose=False))
        return ca.Function('track_tangent', [sym_s], [track_angle], options)

    def get_local_to_global_casadi_fn(self):
        sym_s = ca.MX.sym('s', 1)
        sym_ey = ca.MX.sym('ey', 1)
        sym_ep = ca.MX.sym('ep', 1)

        # Makes sure s is within [0, track_length]
        sym_s_bar = ca.fmod(ca.fmod(sym_s, self.track_length) + self.track_length, self.track_length)

        xy = ca.vertcat(self.x(sym_s_bar), self.y(sym_s_bar))
        dxy = ca.vertcat(self.dx(sym_s_bar), self.dy(sym_s_bar))
        n = dxy / ca.norm_2(dxy)
        n_t = ca.vertcat(-n[1], n[0])

        xy = xy + sym_ey * n_t
        x = xy[0]
        y = xy[1]

        track_angle = ca.atan2(n[1], n[0])
        psi = sym_ep + track_angle

        return ca.Function('local_to_global', [ca.vertcat(sym_s, sym_ey, sym_ep)], [ca.vertcat(x, y, psi)])

    def get_halfwidth(self):
        return self.half_width
    
    def get_track_segment(self, segment_limits: list, resample_resolution: int = None):
        if segment_limits[0] < self.s_waypoints[0]:
            print(f'Track segment start set to {self.s_waypoints[0]}')
            segment_limits[0] = self.s_waypoints[0]
        if segment_limits[1] > self.s_waypoints[-1]:
            print(f'Track segment end set to {self.s_waypoints[-1]}')
            segment_limits[1] = self.s_waypoints[-1]

        if resample_resolution:
            s_waypoints = np.linspace(segment_limits[0], segment_limits[1], int(resample_resolution*(segment_limits[1]-segment_limits[0])))
            xy_waypoints = np.array([np.array(self.x(s_waypoints)).squeeze(), np.array(self.y(s_waypoints)).squeeze()]).T
            left_widths = np.array(self.left_width(s_waypoints)).squeeze()
            right_widths = np.array(self.right_width(s_waypoints)).squeeze()
        else:
            idxs = np.where(np.logical_and(self.s_waypoints >= segment_limits[0], self.s_waypoints <= segment_limits[1]))[0]
            xy_waypoints = self.xy_waypoints[idxs]
            s_waypoints = self.s_waypoints[idxs]
            left_widths = self.left_width_points[idxs]
            right_widths = self.right_width_points[idxs]
        s_waypoints = s_waypoints - s_waypoints[0]

        return CasadiBSplineTrack(xy_waypoints, left_widths, right_widths, self.slack, s_waypoints)

    """
    Coordinate transformation from inertial reference frame (x, y, psi) to curvilinear reference frame (s, e_y, e_psi)
    Input:
        (x, y, psi): position in the inertial reference frame
    Output:
        (s, e_y, e_psi): position in the curvilinear reference frame
    """

    def global_to_local(self, xy_coord):
        x, y, psi = xy_coord
        xy = np.array([x, y])

        s = self.project_to_centerline(xy)

        _dxy = np.array([float(self.dx(s)), float(self.dy(s))])
        n = _dxy / np.linalg.norm(_dxy)
        psi_track = np.arctan2(n[1], n[0])
        epsi = psi - psi_track

        _xy = np.array([float(self.x(s)), float(self.y(s))])
        ey = np.dot(np.array([-np.sin(psi_track), np.cos(psi_track)]), xy - _xy)

        return s, ey, epsi

    def local_to_global(self, cl_coord):
        s, ey, epsi = cl_coord

        xy = np.array([float(self.x(s)), float(self.y(s))])
        dxy = np.array([float(self.dx(s)), float(self.dy(s))])
        n = dxy / np.linalg.norm(dxy)
        n_t = np.array([-n[1], n[0]])

        xy = xy + ey * n_t
        x = xy[0]
        y = xy[1]

        psi_track = np.arctan2(n[1], n[0])
        psi = epsi + psi_track

        return x, y, psi

def plot_tests():
    from mpclab_common.track import get_track
    import matplotlib.pyplot as plt

    rng = np.random.default_rng()

    track = get_track('L_track_barc')
    
    # Compute spline approximation of track
    S = np.linspace(0, track.track_length-1e-3, 200)
    ds = S[1]-S[0]
    for i in range(len(S)-2):
        S[i+1] += rng.random(1)*ds - (ds/2)
    X, Y = [], []
    for s in S:
        # Centerline
        x, y, _ = track.local_to_global((s, 0, 0))
        X.append(x)
        Y.append(y)

    xy = np.array([X, Y]).T
    # print(track.track_length)
    # print(S)

    spline_track = CasadiBSplineTrack(xy, track.track_width, track.slack, s_waypoints=S)
    # spline_track = CasadiBSplineTrack(xy, track.track_width, track.slack, t_waypoints=S/S[-1])
    
    # print(spline_track.track_length)
    # print(spline_track.s_waypoints)   

    cur = []
    spl_cur = []
    for _s in np.linspace(0, track.track_length, 10000):
        cur.append(track.get_curvature(_s))
        spl_cur.append(spline_track.get_curvature(_s))
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(cur, 'b')
    ax.plot(spl_cur, 'g')

    s = rng.random() * S[-1]
    ey = rng.random() * track.track_width - track.track_width/2
    epsi = rng.random() * np.pi/2 - np.pi/4

    x, y, psi = track.local_to_global((s, ey, epsi))
    print(x, y, psi)
    print(spline_track.local_to_global((s, ey, epsi)))

    print(s, ey, epsi)
    print(spline_track.global_to_local((x, y, psi)))

    # fig = plt.figure()
    # ax = fig.gca()
    # spline_track.plot_map(ax)
    # ax.set_aspect('equal')

    # fig = plt.figure()
    # ax = fig.gca()
    # track.plot_map(ax)
    # ax.set_aspect('equal')

    plt.show()

def test_f1():
    import csv
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.transforms import Affine2D

    import pdb

    from mpclab_common.track import get_track, load_tum_raceline

    track = get_track('f1_austin_tenth_scale').get_track_segment([60, 80], 10)

    path = '/home/edward-zhu/Repositories/global_racetrajectory_optimization/outputs/traj_race_cl.csv'
    raceline, s2t, raceline_mat = load_tum_raceline(path, 'f1_austin_tenth_scale', tenth_scale=True, segment=[60, 80])

    fig = plt.figure(figsize=(30,15))
    ax = fig.gca()
    track.plot_map(ax)

    raceline_xy = []
    for s in np.linspace(0, track.track_length, 200):
        raceline_xy.append(np.array(raceline(s2t(s))).squeeze()[:2])
    raceline_xy = np.array(raceline_xy)
    ax.plot(raceline_xy[:,0], raceline_xy[:,1], 'r')

    ax.set_aspect('equal')

    VL = 0.57
    VW = 0.2

    car_rect = patches.Rectangle((-0.5*VL, -0.5*VW), VL, VW, linestyle='solid', color='b', alpha=0.5)

    car_pose = np.array(raceline(s2t(7)))
    b_left = car_pose[0] - VL/2
    b_bot  = car_pose[1] - VW/2
    r = Affine2D().rotate_around(car_pose[0], car_pose[1], car_pose[2]) + ax.transData
    car_rect.set_xy((b_left,b_bot))
    car_rect.set_transform(r)

    ax.add_patch(car_rect)

    # s = np.linspace(0, track.track_length, 1000)
    # l = []
    # r = []
    # c = []
    # x = []
    # y = []
    # dx = []
    # dy = []
    # for _s in s:
    #     c.append(track.get_curvature(_s))
    #     l.append(float(track.left_width(_s)))
    #     r.append(float(track.right_width(_s)))
    #     x.append(float(track.x(_s)))
    #     y.append(float(track.y(_s)))
    #     dx.append(float(track.dx(_s)))
    #     dy.append(float(track.dy(_s)))
    # fig = plt.figure()
    # # ax = fig.gca()
    # # ax.plot(s, l)
    # # ax.plot(s, r)
    # ax = fig.add_subplot(4, 1, 1)
    # ax.plot(s, x)
    # ax = fig.add_subplot(4, 1, 2)
    # ax.plot(s, y)
    # ax = fig.add_subplot(4, 1, 3)
    # ax.plot(s, dx)
    # ax = fig.add_subplot(4, 1, 4)
    # ax.plot(s, dy)

    # track_segment.plot_map(ax)
    # ax.set_aspect('equal')

    plt.show()

if __name__ == "__main__":
    # plot_tests()
    test_f1()
