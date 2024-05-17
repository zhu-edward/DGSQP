#!/usr/bin/env python3

import numpy as np
from numpy import linalg as la
import casadi as ca

import copy
import pdb

class RadiusArclengthTrack():
    def __init__(self, track_width=None, slack=None, cl_segs=None):
        self.track_width = track_width
        self.slack = slack
        self.cl_segs = cl_segs

        self.n_segs = None

        self.key_pts = None
        self.track_length = None

        self.left_width = None
        self.right_width = None

        self.track_extents = None

        self.phase_out = False

        self.circuit = False

    def initialize(self, track_width=None, slack=None, cl_segs=None, init_pos=(0, 0, 0)):
        if track_width is not None:
            self.track_width = track_width
        if slack is not None:
            self.slack = slack
        if cl_segs is not None:
            self.cl_segs = cl_segs

        self.half_width = self.track_width / 2
        self.n_segs = self.cl_segs.shape[0]
        self.key_pts = self.get_track_key_pts(self.cl_segs, init_pos)
        self.track_length = self.key_pts[-1, 3]

        seg_x = self.key_pts[:, 0]
        seg_y = self.key_pts[:, 1]
        seg_t = self.key_pts[:, 2]
        cum_l = self.key_pts[:, 3]
        seg_l = self.key_pts[:, 4]
        seg_c = self.key_pts[:, 5]

        if np.isclose(seg_x[0], seg_x[-1]) and np.isclose(seg_y[0], seg_y[-1]):
            self.circuit = True
            
        # Create casadi lookup tables for the boundary values for each constant curvature segment
        sym_s = ca.SX.sym('s', 1)

        s_bar = ca.fmod(ca.fmod(sym_s, self.track_length) + self.track_length, self.track_length)
        self.left_width = ca.Function('left_width', [sym_s], [ca.pw_lin(s_bar, self.key_pts[1:-1, 3], self.half_width*np.ones(self.key_pts.shape[0]-2))])
        self.right_width = ca.Function('right_width', [sym_s], [ca.pw_lin(s_bar, self.key_pts[1:-1, 3], self.half_width*np.ones(self.key_pts.shape[0]-2))])

        x0 = ca.pw_const(ca.fmod(sym_s, self.track_length), cum_l[1:-1], seg_x[:-1])
        y0 = ca.pw_const(ca.fmod(sym_s, self.track_length), cum_l[1:-1], seg_y[:-1])

        x1 = ca.pw_const(ca.fmod(sym_s, self.track_length), cum_l[1:-1], seg_x[1:])
        y1 = ca.pw_const(ca.fmod(sym_s, self.track_length), cum_l[1:-1], seg_y[1:])
        
        self.segment_start_xy = ca.Function('segment_start_xy', [sym_s], [ca.vertcat(x0, y0)])
        self.segment_end_xy = ca.Function('segment_end_xy', [sym_s], [ca.vertcat(x1, y1)])

        # Cumulative track heading
        cum_t = [seg_t[0]]
        for i in range(self.key_pts.shape[0]-1):
            cum_t.append(cum_t[-1] + seg_l[i+1]*seg_c[i+1])
        cum_t = np.array(cum_t)

        t0 = ca.pw_const(ca.fmod(sym_s, self.track_length), cum_l[1:-1], cum_t[:-1])
        t1 = ca.pw_const(ca.fmod(sym_s, self.track_length), cum_l[1:-1], cum_t[1:])
        
        self.segment_start_t = ca.Function('segment_start_t', [sym_s], [t0])
        self.segment_end_t = ca.Function('segment_end_t', [sym_s], [t1])

        # Cumulative change in track heading
        cum_dt = cum_t - cum_t[0]

        dt0 = ca.pw_const(ca.fmod(sym_s, self.track_length), cum_l[1:-1], cum_dt[:-1])
        dt1 = ca.pw_const(ca.fmod(sym_s, self.track_length), cum_l[1:-1], cum_dt[1:])
        
        self.segment_start_dt = ca.Function('segment_start_dt', [sym_s], [dt0])
        self.segment_end_dt = ca.Function('segment_end_dt', [sym_s], [dt1])

        # Cumulative track progress
        l0 = ca.pw_const(ca.fmod(sym_s, self.track_length), cum_l[1:-1], cum_l[:-1])
        l1 = ca.pw_const(ca.fmod(sym_s, self.track_length), cum_l[1:-1], cum_l[1:])

        self.segment_start_l = ca.Function('segment_start_l', [sym_s], [l0])
        self.segment_end_l = ca.Function('segment_end_l', [sym_s], [l1])

        # Unsigned curvature
        abs_c = ca.pw_const(ca.fmod(sym_s, self.track_length), cum_l[1:-1], np.abs(seg_c[1:]))
        
        self.segment_abs_c = ca.Function('segment_abs_c', [sym_s], [abs_c])
        
        # Get the x-y extents of the track
        s_grid = np.linspace(0, self.track_length, int(10 * self.track_length))
        x_grid, y_grid = [], []
        for s in s_grid:
            xp, yp, _ = self.local_to_global((s, self.half_width + self.slack, 0))
            xm, ym, _ = self.local_to_global((s, -self.half_width - self.slack, 0))
            x_grid.append(xp)
            x_grid.append(xm)
            y_grid.append(yp)
            y_grid.append(ym)
        self.track_extents = dict(x_min=np.amin(x_grid), x_max=np.amax(x_grid), y_min=np.amin(y_grid),
                                  y_max=np.amax(y_grid))

        spline_options = dict(degree=[3])
        # Compute spline approximation of track
        self.s_waypoints = np.linspace(0, self.track_length-1e-3, 100)
        X, Y = [], []
        for s in self.s_waypoints:
            # Centerline
            x, y, _ = self.local_to_global((s, 0, 0))
            X.append(x)
            Y.append(y)
        self.xy_waypoints = np.array([X, Y]).T
        self.x = ca.interpolant('x_spline', 'bspline', [self.s_waypoints], self.xy_waypoints[:,0], spline_options)
        self.y = ca.interpolant('y_spline', 'bspline', [self.s_waypoints], self.xy_waypoints[:,1], spline_options)
        # First and second derivatives of position w.r.t. s
        s_sym = ca.MX.sym('s', 1)
        self.dx = ca.Function('dx', [s_sym], [ca.jacobian(self.x(s_sym), s_sym)])
        self.dy = ca.Function('dy', [s_sym], [ca.jacobian(self.y(s_sym), s_sym)])
        self.ddx = ca.Function('ddx', [s_sym], [ca.jacobian(self.dx(s_sym), s_sym)])
        self.ddy = ca.Function('ddy', [s_sym], [ca.jacobian(self.dy(s_sym), s_sym)])
        # Spline approximation of track boundaries
        xi, yi, xo, yo = [], [], [], []
        for s in self.s_waypoints:
            _xi, _yi, _ = self.local_to_global((s, float(self.left_width(s)), 0))
            xi.append(_xi)
            yi.append(_yi)
            _xo, _yo, _ = self.local_to_global((s, -float(self.right_width(s)), 0))
            xo.append(_xo)
            yo.append(_yo)
        self.xi = ca.interpolant('xi_s', 'bspline', [self.s_waypoints], xi)
        self.yi = ca.interpolant('yi_s', 'bspline', [self.s_waypoints], yi)
        self.xo = ca.interpolant('xo_s', 'bspline', [self.s_waypoints], xo)
        self.yo = ca.interpolant('yo_s', 'bspline', [self.s_waypoints], yo)

        # NLP for global to local conversion
        s_sym = ca.MX.sym('s', 1)
        xy_sym = ca.MX.sym('xy', 2)
        xy = ca.vertcat(self.x(s_sym), self.y(s_sym))
        objective = ca.bilin(np.eye(2), xy_sym - xy, xy_sym - xy)
        prob = {'x': s_sym, 'f': objective, 'p': xy_sym}
        ipopt_opts = dict(print_level=0,
                          linear_solver='ma27')
        solver_opts = dict(error_on_fail=False, 
                        ipopt=ipopt_opts, 
                        verbose=False, 
                        print_time=False, 
                        verbose_init=False)
        self.global_to_local_solver = ca.nlpsol('g2l', 'ipopt', prob, solver_opts)

        return

    def global_to_local_typed(self, data):  # data is vehicleState
        xy_coord = (data.x.x, data.x.y, data.e.psi)
        cl_coord = self.global_to_local(xy_coord)
        if cl_coord:
            data.p.s = cl_coord[0]
            data.p.x_tran = cl_coord[1]
            data.p.e_psi = cl_coord[2]
            return 0
        return -1

    def local_to_global_typed(self, data):
        cl_coord = (data.p.s, data.p.x_tran, data.p.e_psi)
        xy_coord = self.local_to_global(cl_coord)
        if xy_coord:
            data.x.x = xy_coord[0]
            data.x.y = xy_coord[1]
            data.e.psi = xy_coord[2]
            return -1
        return 0

    def get_curvature(self, s):
        # while s < 0: s += self.track_length
        # while s >= self.track_length: s -= self.track_length
        s = np.mod(np.mod(s, self.track_length) + self.track_length, self.track_length)

        # Find key point indicies corresponding to current segment
        # key_pts = [x, y, psi, cumulative length, segment length, signed curvature]
        key_pt_idx_s = np.where(s >= self.key_pts[:, 3])[0][-1]

        return self.key_pts[key_pt_idx_s + 1, 5]  # curvature at this keypoint

    def update_curvature(self, state):
        for i in range(int(state.lookahead.n)):
            state.lookahead.curvature[i] = self.get_curvature(state.p.s + state.lookahead.dl * i)

    def get_curvature_casadi_fn(self):
        sym_s = ca.SX.sym('s', 1)
        # Makes sure s is within [0, track_length]
        sym_s_bar = ca.fmod(ca.fmod(sym_s, self.track_length) + self.track_length, self.track_length)
        # Piecewise constant function mapping s to track curvature
        pw_const_curvature = ca.pw_const(sym_s_bar, self.key_pts[1:-1, 3], self.key_pts[1:, 5])
        return ca.Function('track_curvature', [sym_s], [pw_const_curvature])

    def get_tangent_angle_casadi_fn(self):        
        seg_len = copy.copy(self.key_pts[:, 4])
        curvature = copy.copy(self.key_pts[:, 5])

        abs_angs = np.zeros(self.key_pts.shape[0] + 1)
        for i in range(self.key_pts.shape[0]):
            if curvature[i] == 0:
                abs_angs[i + 1] = abs_angs[i]
            else:
                abs_angs[i + 1] = abs_angs[i] + seg_len[i] * curvature[i]
        abs_angs = abs_angs[1:]

        sym_s = ca.SX.sym('s', 1)
        # Makes sure s is within [0, track_length]
        sym_s_bar = ca.fmod(ca.fmod(sym_s, self.track_length) + self.track_length, self.track_length)
        # Piecewise linear function mapping s to track tangent angle
        pw_lin_tangent_ang = ca.pw_lin(sym_s_bar, self.key_pts[:, 3], abs_angs)

        return ca.Function('track_tangent', [sym_s], [pw_lin_tangent_ang])

    """
    Retuns a differentiable CasADi function which performs the coordinate transformation from curvilinear reference frame (s, e_y, e_psi) to inertial reference frame (x, y, psi)
    """
    def get_local_to_global_casadi_fn(self):
        sym_s = ca.SX.sym('s', 1)
        sym_ey = ca.SX.sym('ey', 1)
        sym_ep = ca.SX.sym('ep', 1)

        # Makes sure s is within [0, track_length]
        sym_s_bar = ca.fmod(ca.fmod(sym_s, self.track_length) + self.track_length, self.track_length)
        
        tangent_angle = self.get_tangent_angle_casadi_fn()

        # Global heading
        p = sym_ep + tangent_angle(sym_s_bar)
        
        # Get the position and heading of the start of the segment containing s
        xy_ = self.segment_start_xy(sym_s_bar)
        t_ = self.segment_start_t(sym_s_bar)
        l_ = self.segment_start_l(sym_s_bar)
        
        dt = self.segment_end_t(sym_s_bar) - t_
        dl = self.segment_end_l(sym_s_bar) - l_
        
        # Change in x and y due to s
        t_bar = dt*(sym_s_bar-l_)/dl
        # Compute the chord length when nonzero curvature or linear distance when zero curvature
        ch = ca.if_else(dt == 0, sym_s_bar-l_, (1/self.segment_abs_c(sym_s_bar))*ca.sqrt(2*(1-ca.cos(t_bar))))
        dx_s = ch*ca.cos(t_bar/2 + t_)
        dy_s = ch*ca.sin(t_bar/2 + t_)

        # Change in x and y due to ey
        dx_ey = sym_ey*ca.cos(t_bar + ca.pi/2 + t_)
        dy_ey = sym_ey*ca.sin(t_bar + ca.pi/2 + t_)

        x = xy_[0] + dx_s + dx_ey
        y = xy_[1] + dy_s + dy_ey

        return ca.Function('local_to_global', [ca.vertcat(sym_s, sym_ey, sym_ep)], [ca.vertcat(x, y, p)])

    """
    Compute the index corresponding to the closest waypoint w.r.t. the x-y position
    """
    def get_closest_waypoint_index(self, xy):
        dist = []
        for _xy in self.xy_waypoints:
            dist.append(np.linalg.norm(xy - _xy))
        idx = np.argmin(dist)
        return idx
    
    """
    Returns the track progress 's' for a given x-y position 'xy'
    """
    def project_to_centerline(self, xy):
        _i = self.get_closest_waypoint_index(xy)
        sol = self.global_to_local_solver(x0=self.s_waypoints[_i], lbx=0, ubx=self.track_length, p=xy)
        success = self.global_to_local_solver.stats()['success']
        if not success:
            raise(ValueError(self.global_to_local_solver.stats()['return_status']))
        return float(sol['x'])
    
    """
    Returns a differentiable CasADi function which is equivalent to self.project_to_centerline 
    """
    def get_progress_projection_casadi_fn(self):
        class progress_projection(ca.Callback):
            def __init__(self, name, track, opts={}):
                ca.Callback.__init__(self)
                self.construct(name, opts)
                self.track = track

            # Number of inputs and outputs
            def get_n_in(self): return 1
            def get_n_out(self): return 1

            def get_sparsity_in(self, i):
                return ca.Sparsity.dense(2, 1)

            def get_sparsity_out(self, i):
                return ca.Sparsity.dense(1, 1)

            # Initialize the object
            def init(self):
                pass

            # Evaluate numerically
            def eval(self, arg):
                xy = arg[0]
                return [self.track.project_to_centerline(xy)]
            
            def has_jacobian(self): return True

            def get_jacobian(self, name, inames, onames, opts):
                class JacFun(ca.Callback):
                    def __init__(self, track, opts={'enable_fd': True}):
                        ca.Callback.__init__(self)
                        self.construct(name, opts)
                        self.track = track

                    # Gradient of optimal 's' w.r.t. position 'p
                    def dsdp(self, p):
                        s_sol = self.track.project_to_centerline(p)
                        t = ca.vertcat(self.track.x(s_sol), self.track.y(s_sol))
                        dt = ca.vertcat(self.track.dx(s_sol), self.track.dy(s_sol))
                        d2t = ca.vertcat(self.track.ddx(s_sol), self.track.ddy(s_sol))
                        dsdp = dt/(ca.norm_2(dt)**2 - ca.dot(p-t, d2t))
                        return dsdp.T
            
                    def get_n_in(self): return 2
                    def get_n_out(self): return 1

                    def get_sparsity_in(self, i):
                        if i == 0:
                            return ca.Sparsity.dense(2, 1)
                        elif i == 1:
                            return ca.Sparsity.dense(1, 1)
                    
                    def get_sparsity_out(self, i):
                        return ca.Sparsity.dense(1, 2)

                    # Evaluate numerically
                    def eval(self, arg):
                        xy = arg[0]
                        return [self.dsdp(xy)]

                # You are required to keep a reference alive to the returned Callback object
                self.jac_callback = JacFun(self.track)
                return self.jac_callback
        
        return progress_projection('progress_projection', self)
    
    def get_halfwidth(self, s):
        return self.half_width

    def get_track_key_pts(self, cl_segs, init_pos):
        if cl_segs is None:
            raise ValueError('Track segments have not been defined')

        n_segs = cl_segs.shape[0]
        # Given the segments in cl_segs we compute the (x, y) points of the track and the angle of the tangent vector (psi) at
        # these points. For each segment we compute the (x, y, psi) coordinate at the last point of the segment. Furthermore,
        # we compute also the cumulative length at the starting point of the segment at signed curvature
        # key_pts = [x, y, psi, cumulative length, segment length, signed curvature]
        track_key_pts = np.zeros((n_segs + 1, 6))
        track_key_pts[0, 0] = init_pos[0]
        track_key_pts[0, 1] = init_pos[1]
        track_key_pts[0, 2] = init_pos[2]
        for i in range(1, n_segs + 1):
            x_prev = track_key_pts[i - 1, 0]
            y_prev = track_key_pts[i - 1, 1]
            psi_prev = track_key_pts[i - 1, 2]
            cum_s_prev = track_key_pts[i - 1, 3]

            l = cl_segs[i - 1, 0]
            r = cl_segs[i - 1, 1]

            if r == 0:
                # No curvature (i.e. straight line)
                psi = psi_prev
                x = x_prev + l * np.cos(psi_prev)
                y = y_prev + l * np.sin(psi_prev)
                curvature = 0
            else:
                # dir = np.sign(r)

                # Find coordinates for center of curved segment
                x_c = x_prev - r * (np.sin(psi_prev))
                y_c = y_prev + r * (np.cos(psi_prev))
                # Angle spanned by curved segment
                theta = l / r
                # end of curve
                x = x_c + r * np.sin(psi_prev + theta)
                y = y_c - r * np.cos(psi_prev + theta)
                # curvature of segment
                curvature = 1 / r

                # next angle
                psi = wrap_angle(psi_prev + theta)
            cum_s = cum_s_prev + l
            track_key_pts[i] = np.array([x, y, psi, cum_s, l, curvature])

        return track_key_pts

    def get_track_xy(self, pts_per_dist=None, close_loop=True):
        if self.key_pts is None:
            raise ValueError('Track key points have not been defined')

        if pts_per_dist is None:
            pts_per_dist = 2000 / self.track_length
        
        # Start line
        init_x = self.key_pts[0, 0]
        init_y = self.key_pts[0, 1]
        init_psi = self.key_pts[0, 2]
        start_line_x = [init_x + np.cos(init_psi + np.pi / 2) * self.track_width / 2,
                        init_x - np.cos(init_psi + np.pi / 2) * self.track_width / 2]
        start_line_y = [init_y + np.sin(init_psi + np.pi / 2) * self.track_width / 2,
                        init_y - np.sin(init_psi + np.pi / 2) * self.track_width / 2]

        # Center line and boundaries
        x_track = []
        x_bound_in = []
        x_bound_out = []
        y_track = []
        y_bound_in = []
        y_bound_out = []
        for i in range(1, self.key_pts.shape[0]):
            l = self.key_pts[i, 4]
            cum_s = self.key_pts[i - 1, 3]
            n_pts = np.around(l * pts_per_dist)
            d = np.linspace(0, l, int(n_pts))
            for j in d:
                cl_coord = (j + cum_s, 0, 0)
                xy_coord = self.local_to_global(cl_coord)
                x_track.append(xy_coord[0])
                y_track.append(xy_coord[1])
                cl_coord = (j + cum_s, self.track_width / 2, 0)
                xy_coord = self.local_to_global(cl_coord)
                x_bound_in.append(xy_coord[0])
                y_bound_in.append(xy_coord[1])
                cl_coord = (j + cum_s, -self.track_width / 2, 0)
                xy_coord = self.local_to_global(cl_coord)
                x_bound_out.append(xy_coord[0])
                y_bound_out.append(xy_coord[1])
        if not close_loop:
            x_track = x_track[:-1]
            y_track = y_track[:-1]
            x_bound_in = x_bound_in[:-1]
            y_bound_in = y_bound_in[:-1]
            x_bound_out = x_bound_out[:-1]
            y_bound_out = y_bound_out[:-1]

        D = dict(start=dict(x=start_line_x, y=start_line_y), 
                 center=dict(x=x_track, y=y_track),
                 bound_in=dict(x=x_bound_in, y=y_bound_in),
                 bound_out=dict(x=x_bound_out, y=y_bound_out))
        
        return D
        
    def plot_map(self, ax, pts_per_dist=None, close_loop=True, distance_markers=0, show_segments=False):
        if self.key_pts is None:
            raise ValueError('Track key points have not been defined')

        if pts_per_dist is None:
            pts_per_dist = 2000 / self.track_length

        # Plot the starting line
        init_x = self.key_pts[0, 0]
        init_y = self.key_pts[0, 1]
        init_psi = self.key_pts[0, 2]
        start_line_x = [init_x + np.cos(init_psi + np.pi / 2) * self.track_width / 2,
                        init_x - np.cos(init_psi + np.pi / 2) * self.track_width / 2]
        start_line_y = [init_y + np.sin(init_psi + np.pi / 2) * self.track_width / 2,
                        init_y - np.sin(init_psi + np.pi / 2) * self.track_width / 2]
        ax.plot(start_line_x, start_line_y, 'r', linewidth=1)

        # Plot the track and boundaries
        x_track = []
        x_bound_in = []
        x_bound_out = []
        y_track = []
        y_bound_in = []
        y_bound_out = []
        for i in range(1, self.key_pts.shape[0]):
            l = self.key_pts[i, 4]
            cum_s = self.key_pts[i - 1, 3]
            n_pts = np.around(l * pts_per_dist)
            d = np.linspace(0, l,
                            int(n_pts))  # FIXED - numpy no longer allows interpolation using a double number of points
            for j in d:
                cl_coord = (j + cum_s, 0, 0)
                xy_coord = self.local_to_global(cl_coord)
                x_track.append(xy_coord[0])
                y_track.append(xy_coord[1])
                cl_coord = (j + cum_s, self.track_width / 2, 0)
                xy_coord = self.local_to_global(cl_coord)
                x_bound_in.append(xy_coord[0])
                y_bound_in.append(xy_coord[1])
                cl_coord = (j + cum_s, -self.track_width / 2, 0)
                xy_coord = self.local_to_global(cl_coord)
                x_bound_out.append(xy_coord[0])
                y_bound_out.append(xy_coord[1])

            if show_segments:
                p_i = self.local_to_global((cum_s, self.track_width/2, 0))
                p_o = self.local_to_global((cum_s, -self.track_width/2, 0))
                ax.plot([p_i[0], p_o[0]], [p_i[1], p_o[1]], 'm', linewidth=1)
            
        if close_loop:
            ax.plot(x_track, y_track, 'k--', linewidth=1)
            ax.plot(x_bound_in, y_bound_in, 'k')
            ax.plot(x_bound_out, y_bound_out, 'k')
        else:
            ax.plot(x_track[:-1], y_track[:-1], 'k--', linewidth=1)
            ax.plot(x_bound_in[:-1], y_bound_in[:-1], 'k')
            ax.plot(x_bound_out[:-1], y_bound_out[:-1], 'k')

        if distance_markers > 0:
            if self.track_length >= 1:
                for s in np.arange(distance_markers, self.track_length, distance_markers):
                    p_i = self.local_to_global((s, self.track_width/2, 0))
                    p_o = self.local_to_global((s, -self.track_width/2, 0))
                    ax.plot([p_i[0], p_o[0]], [p_i[1], p_o[1]], 'b', linewidth=1)
                    t = np.arctan2(p_i[1]-p_o[1], p_i[0]-p_o[0])
                    if t >= 0 and t <= np.pi/2:
                        anchor = ('right','top')
                    elif t > np.pi/2 and t <= np.pi:
                        anchor = ('left','top')
                    elif t >= -np.pi/2 and t <= 0:
                        anchor = ('right','bottom')
                    elif t > -np.pi and t <= -np.pi/2:
                        anchor = ('left','bottom')
                    ax.text(p_o[0], p_o[1], str(s), ha=anchor[0], va=anchor[1])

        track_bbox = (np.amin(x_bound_out),
                      np.amin(y_bound_out),
                      np.amax(x_bound_out) - np.amin(x_bound_out),
                      np.amax(y_bound_out) - np.amin(y_bound_out))

        return track_bbox

    def plot_map_qt(self, p, pts_per_dist=None, close_loop=True, show_meter_markers=False):
        import pyqtgraph as pg
        
        if self.key_pts is None:
            raise ValueError('Track key points have not been defined')

        if pts_per_dist is None:
            pts_per_dist = 2000 / self.track_length

        # Plot the starting line
        init_x = self.key_pts[0, 0]
        init_y = self.key_pts[0, 1]
        init_psi = self.key_pts[0, 2]
        start_line_x = [init_x + np.cos(init_psi + np.pi / 2) * self.track_width / 2,
                        init_x - np.cos(init_psi + np.pi / 2) * self.track_width / 2]
        start_line_y = [init_y + np.sin(init_psi + np.pi / 2) * self.track_width / 2,
                        init_y - np.sin(init_psi + np.pi / 2) * self.track_width / 2]

        p.plot(start_line_x, start_line_y, pen=pg.mkPen('r', width=1))

        # Plot the track and boundaries
        x_track = []
        x_bound_in = []
        x_bound_out = []
        y_track = []
        y_bound_in = []
        y_bound_out = []
        for i in range(1, self.key_pts.shape[0]):
            l = self.key_pts[i, 4]
            cum_s = self.key_pts[i - 1, 3]
            n_pts = np.around(l * pts_per_dist)
            d = np.linspace(0, l,
                            int(n_pts))  # FIXED - numpy no longer allows interpolation using a double number of points
            for j in d:
                cl_coord = (j + cum_s, 0, 0)
                xy_coord = self.local_to_global(cl_coord)
                x_track.append(xy_coord[0])
                y_track.append(xy_coord[1])
                cl_coord = (j + cum_s, self.track_width / 2, 0)
                xy_coord = self.local_to_global(cl_coord)
                x_bound_in.append(xy_coord[0])
                y_bound_in.append(xy_coord[1])
                cl_coord = (j + cum_s, -self.track_width / 2, 0)
                xy_coord = self.local_to_global(cl_coord)
                x_bound_out.append(xy_coord[0])
                y_bound_out.append(xy_coord[1])
        if close_loop:
            p.plot(x_track, y_track, pen=pg.mkPen('k', width=1, dash=[4, 2]))
            p.plot(x_bound_in, y_bound_in, pen=pg.mkPen('k', width=1))
            p.plot(x_bound_out, y_bound_out, pen=pg.mkPen('k', width=1))
        else:
            p.plot(x_track[:-1], y_track[:-1], pen=pg.mkPen('k', width=1, dash=[4, 2]))
            p.plot(x_bound_in[:-1], y_bound_in[:-1], pen=pg.mkPen('k', width=1))
            p.plot(x_bound_out[:-1], y_bound_out[:-1], pen=pg.mkPen('k', width=1))

        if show_meter_markers:
            if self.track_length >= 1:
                for s in range(1, int(np.floor(self.track_length))+1):
                    p_i = self.local_to_global((s, self.track_width/2, 0))
                    p_o = self.local_to_global((s, -self.track_width/2, 0))
                    p.plot([p_i[0], p_o[0]], [p_i[1], p_o[1]], pen=pg.mkPen('b', width=1))
                    t = np.arctan2(p_i[1]-p_o[1], p_i[0]-p_o[0])
                    if t >= 0 and t <= np.pi/2:
                        anchor = (1,0)
                    elif t > np.pi/2 and t <= np.pi:
                        anchor = (0,0)
                    elif t >= -np.pi/2 and t <= 0:
                        anchor = (1,1)
                    elif t > -np.pi and t <= -np.pi/2:
                        anchor = (0,1)
                    T = pg.TextItem(str(s), anchor=anchor)
                    p.addItem(T)
                    T.setPos(p_o[0], p_o[1])
                    
        track_bbox = (np.amin(x_bound_out),
                      np.amin(y_bound_out),
                      np.amax(x_bound_out) - np.amin(x_bound_out),
                      np.amax(y_bound_out) - np.amin(y_bound_out))

        return track_bbox

    def remove_phase_out(self):
        if self.phase_out:
            self.track_length = self.key_pts[-2][3]
            self.key_pts = self.key_pts[0:-1]
            self.n_segs = self.n_segs - 1
            self.cl_segs = self.cl_segs[0:-1]
            self.phase_out = False

    """
    Coordinate transformation from inertial reference frame (x, y, psi) to curvilinear reference frame (s, e_y, e_psi)
    Input:
        (x, y, psi): position in the inertial reference frame
    Output:
        (s, e_y, e_psi): position in the curvilinear reference frame
    """
    def global_to_local(self, xy_coord, line='center'):
        if self.key_pts is None:
            raise ValueError('Track key points have not been defined')

        x, y, psi = xy_coord

        pos_cur = np.array([x, y])
        cl_coord = None

        for i in range(1, self.key_pts.shape[0]):
            # key_pts = [x, y, psi, cumulative length, segment length, signed curvature]
            x_s = self.key_pts[i - 1, 0]
            y_s = self.key_pts[i - 1, 1]
            psi_s = self.key_pts[i - 1, 2]
            curve_s = self.key_pts[i - 1, 5]
            x_f = self.key_pts[i, 0]
            y_f = self.key_pts[i, 1]
            psi_f = self.key_pts[i, 2]
            curve_f = self.key_pts[i, 5]

            l = self.key_pts[i, 4]

            pos_s = np.array([x_s, y_s])
            pos_f = np.array([x_f, y_f])

            # Check if at any of the segment start or end points
            if la.norm(pos_s - pos_cur) == 0:
                # At start of segment
                s = self.key_pts[i - 1, 3]
                e_y = 0
                e_psi = np.unwrap([psi_s, psi])[1] - psi_s
                cl_coord = (s, e_y, e_psi)
                break
            if la.norm(pos_f - pos_cur) == 0:
                # At end of segment
                s = self.key_pts[i, 3]
                e_y = 0
                e_psi = np.unwrap([psi_f, psi])[1] - psi_f
                cl_coord = (s, e_y, e_psi)
                break

            if curve_f == 0:
                # Check if on straight segment
                if np.abs(compute_angle(pos_s, pos_cur, pos_f)) <= np.pi / 2 and np.abs(
                        compute_angle(pos_f, pos_cur, pos_s)) <= np.pi / 2:
                    v = pos_cur - pos_s
                    ang = compute_angle(pos_s, pos_f, pos_cur)
                    e_y = la.norm(v) * np.sin(ang)
                    # Check if deviation from centerline is within track width plus some slack for current segment
                    # (allows for points outside of track boundaries)
                    if np.abs(e_y) <= self.track_width / 2 + self.slack:
                        d = la.norm(v) * np.cos(ang)
                        s = self.key_pts[i - 1, 3] + d
                        e_psi = np.unwrap([psi_s, psi])[1] - psi_s
                        cl_coord = (s, e_y, e_psi)
                        break
                    else:
                        continue
                else:
                    continue
            else:
                # Check if on curved segment
                r = 1 / curve_f
                dir = np.sign(r)

                # Find coordinates for center of curved segment
                x_c = x_s + np.abs(r) * np.cos(psi_s + dir * np.pi / 2)
                y_c = y_s + np.abs(r) * np.sin(psi_s + dir * np.pi / 2)
                curve_center = np.array([x_c, y_c])

                span_ang = l / r
                cur_ang = compute_angle(curve_center, pos_s, pos_cur)
                if np.sign(span_ang) == np.sign(cur_ang) and np.abs(span_ang) >= np.abs(cur_ang):
                    v = pos_cur - curve_center
                    e_y = -np.sign(dir) * (la.norm(v) - np.abs(r))
                    # Check if deviation from centerline is within track width plus some slack for current segment
                    # (allows for points outside of track boundaries)
                    if np.abs(e_y) <= self.track_width / 2 + self.slack:
                        d = np.abs(cur_ang) * np.abs(r)
                        s = self.key_pts[i - 1, 3] + d
                        e_psi = np.unwrap([psi_s + cur_ang, psi])[1] - (psi_s + cur_ang)
                        cl_coord = (s, e_y, e_psi)
                        break
                    else:
                        continue
                else:
                    continue

        if line == 'inside':
            cl_coord = (cl_coord[0], cl_coord[1] - self.track_width / 5, cl_coord[2])
        elif line == 'outside':
            cl_coord = (cl_coord[0], cl_coord[1] + self.track_width / 5, cl_coord[2])
        elif line == 'pid_offset':
            # PID controller tends to cut to the inside of the track
            cl_coord = (cl_coord[0], cl_coord[1] + (0.1 * self.track_width / 2), cl_coord[2])

        # if cl_coord is None:
        #     raise ValueError('Point is out of the track!')

        return cl_coord
    
    """
    Coordinate transformation from curvilinear reference frame (s, e_y, e_psi) to inertial reference frame (x, y, psi)
    Input:
        (s, e_y, e_psi): position in the curvilinear reference frame
    Output:
        (x, y, psi): position in the inertial reference frame
    """
    def local_to_global(self, cl_coord):
        if self.key_pts is None:
            raise ValueError('Track key points have not been defined')

        # s = np.mod(cl_coord[0], self.track_length) # Distance along current lap
        s = cl_coord[0]
        while s < 0: s += self.track_length
        while s >= self.track_length: s -= self.track_length

        e_y = cl_coord[1]
        e_psi = cl_coord[2]

        # Find key point indicies corresponding to current segment
        # key_pts = [x, y, psi, cumulative length, segment length, signed curvature]
        key_pt_idx_s = np.where(s >= self.key_pts[:, 3])[0][-1]
        key_pt_idx_f = key_pt_idx_s + 1
        seg_idx = key_pt_idx_s

        x_s = self.key_pts[key_pt_idx_s, 0]
        y_s = self.key_pts[key_pt_idx_s, 1]
        psi_s = self.key_pts[key_pt_idx_s, 2]
        curve_s = self.key_pts[key_pt_idx_s, 5]
        x_f = self.key_pts[key_pt_idx_f, 0]
        y_f = self.key_pts[key_pt_idx_f, 1]
        psi_f = self.key_pts[key_pt_idx_f, 2]
        curve_f = self.key_pts[key_pt_idx_f, 5]

        l = self.key_pts[key_pt_idx_f, 4]
        d = s - self.key_pts[key_pt_idx_s, 3]  # Distance along current segment

        if curve_f == 0:
            # Segment is a straight line
            x = x_s + (x_f - x_s) * d / l + e_y * np.cos(psi_f + np.pi / 2)
            y = y_s + (y_f - y_s) * d / l + e_y * np.sin(psi_f + np.pi / 2)
            psi = wrap_angle(psi_f + e_psi)
        else:
            r = 1 / curve_f
            dir = sign(r)

            # Find coordinates for center of curved segment
            x_c = x_s + np.abs(r) * np.cos(psi_s + dir * np.pi / 2)
            y_c = y_s + np.abs(r) * np.sin(psi_s + dir * np.pi / 2)

            # Angle spanned up to current location along segment
            span_ang = d / np.abs(r)

            # Angle of the tangent vector at the current location
            psi_d = wrap_angle(psi_s + dir * span_ang)

            ang_norm = wrap_angle(psi_s + dir * np.pi / 2)
            ang = -sign(ang_norm) * (np.pi - np.abs(ang_norm))

            x = x_c + (np.abs(r) - dir * e_y) * np.cos(ang + dir * span_ang)
            y = y_c + (np.abs(r) - dir * e_y) * np.sin(ang + dir * span_ang)
            psi = wrap_angle(psi_d + e_psi)
        return (x, y, psi)

def wrap_angle(theta):
    if theta < -np.pi:
        wrapped_angle = 2 * np.pi + theta
    elif theta > np.pi:
        wrapped_angle = theta - 2 * np.pi
    else:
        wrapped_angle = theta

    return wrapped_angle


def sign(a):
    if a >= 0:
        res = 1
    else:
        res = -1

    return res


"""
Helper function for computing the angle between the vectors point_1-point_0
and point_2-point_0. All points are defined in the inertial frame
Input:
    point_0: position of the intersection point (np.array of size 2)
    point_1, point_2: defines the intersecting lines (np.array of size 2)
Output:
    theta: angle in radians
"""
def compute_angle(point_0, point_1, point_2):
    v_1 = point_1 - point_0
    v_2 = point_2 - point_0

    dot = v_1.dot(v_2)
    det = v_1[0] * v_2[1] - v_1[1] * v_2[0]
    theta = np.arctan2(det, dot)

    return theta

if __name__ == "__main__":
    from mpclab_common.track import get_track

    track = get_track('L_track_barc')

    f = track.get_progress_projection_casadi_fn()

    print(f([1,0]))

    p = ca.MX.sym('p', 2)
    df = ca.Function('df', [p], [ca.jacobian(f(p), p)])

    print(df([1,0]))

    pdb.set_trace()
    # surf = track.get_barc3d_surface()
