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

