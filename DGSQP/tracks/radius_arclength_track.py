#!/usr/bin/env python3

import numpy as np
from numpy import linalg as la

import casadi as ca
import pyqtgraph as pg

import copy

class RadiusArclengthTrack():
    def __init__(self, track_width=None, slack=None, cl_segs=None):
        self.track_width = track_width
        self.slack = slack
        self.cl_segs = cl_segs

        self.n_segs = None

        self.key_pts = None
        self.track_length = None

        self.track_extents = None

        self.phase_out = False
        self.circuit = True # Flag for whether the track is a circuit

    def initialize(self, track_width=None, slack=None, cl_segs=None, init_pos=(0,0,0)):
        if track_width is not None:
            self.track_width = track_width
        if slack is not None:
            self.slack     = slack
        if cl_segs is not None:
            self.cl_segs = cl_segs

        self.half_width = self.track_width/2
        self.n_segs = self.cl_segs.shape[0]
        self.key_pts = self.get_track_key_pts(self.cl_segs, init_pos)
        self.track_length = self.key_pts[-1,3]

        # Get the x-y extents of the track
        s_grid = np.linspace(0, self.track_length, int(10*self.track_length))
        x_grid, y_grid = [], []
        for s in s_grid:
            xp, yp, _ = self.local_to_global((s, self.half_width+self.slack, 0))
            xm, ym, _ = self.local_to_global((s, -self.half_width-self.slack, 0))
            x_grid.append(xp)
            x_grid.append(xm)
            y_grid.append(yp)
            y_grid.append(ym)
        self.track_extents = dict(x_min=np.amin(x_grid), x_max=np.amax(x_grid), y_min=np.amin(y_grid), y_max=np.amax(y_grid))
        return

    def global_to_local_typed(self,data): # data is vehicleState
        xy_coord = (data.x.x, data.x.y, data.e.psi)
        cl_coord = self.global_to_local(xy_coord)
        if cl_coord:
            data.p.s = cl_coord[0]
            data.p.x_tran = cl_coord[1]
            data.p.e_psi = cl_coord[2]
            return 0
        return -1

    def local_to_global_typed(self,data):
        cl_coord = (data.p.s, data.p.x_tran, data.p.e_psi)
        xy_coord = self.local_to_global(cl_coord)
        if xy_coord:
            data.x.x = xy_coord[0]
            data.x.y = xy_coord[1]
            data.e.psi = xy_coord[2]
            return -1
        return 0

    def get_curvature(self, s):
        # Find key point indicies corresponding to current segment
        # key_pts = [x, y, psi, cumulative length, segment length, signed curvature]

        while s < 0: s += self.track_length
        while s >= self.track_length: s -= self.track_length
        key_pt_idx_s = np.where(s >= self.key_pts[:,3])[0][-1]
        #d = s - self.key_pts[key_pt_idx_s, 3] # Distance along current segment

        return self.key_pts[key_pt_idx_s + 1, 5] # curvature at this keypoint

    def get_curvature_casadi_fn(self):
        sym_s = ca.SX.sym('s', 1)
        # Makes sure s is within [0, track_length]
        sym_s_bar = ca.mod(ca.mod(sym_s, self.track_length)+self.track_length, self.track_length)
        # Piecewise constant function mapping s to track curvature
        pw_const_curvature = ca.pw_const(sym_s_bar, self.key_pts[1:-1,3], self.key_pts[1:,5])
        return ca.Function('track_curvature', [sym_s], [pw_const_curvature])

    def get_tangent_angle_casadi_fn(self):
        sym_s = ca.SX.sym('s', 1)
        # abs_angs = copy.copy(self.key_pts[:,2])
        seg_len = copy.copy(self.key_pts[:,4])
        curvature = copy.copy(self.key_pts[:,5])

        abs_angs = np.zeros(self.key_pts.shape[0]+1)
        for i in range(self.key_pts.shape[0]):
            if curvature[i] == 0:
                abs_angs[i+1] = abs_angs[i]
            else:
                abs_angs[i+1] = abs_angs[i] + seg_len[i]*curvature[i]
        abs_angs = abs_angs[1:]
        # if self.circuit:
        #     abs_angs[-2:] = abs_angs[0] + 2*np.pi # Assumes that the last track segment is straight
        
        # Makes sure s is within [0, track_length]
        sym_s_bar = ca.mod(ca.mod(sym_s, self.track_length)+self.track_length, self.track_length)
        # Piecewise linear function mapping s to track tangent angle
        pw_lin_tangent_ang = ca.pw_lin(sym_s_bar, self.key_pts[:,3], abs_angs)
        return ca.Function('track_tangent', [sym_s], [pw_lin_tangent_ang])

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
        track_key_pts = np.zeros((n_segs+1,6))
        track_key_pts[0,0] = init_pos[0]
        track_key_pts[0,1] = init_pos[1]
        track_key_pts[0,2] = init_pos[2]
        for i in range(1, n_segs+1):
            x_prev = track_key_pts[i-1,0]
            y_prev = track_key_pts[i-1,1]
            psi_prev = track_key_pts[i-1,2]
            cum_s_prev = track_key_pts[i-1,3]

            l = cl_segs[i-1,0]
            r = cl_segs[i-1,1]

            if r == 0:
                # No curvature (i.e. straight line)
                psi = psi_prev
                x = x_prev + l*np.cos(psi_prev)
                y = y_prev + l*np.sin(psi_prev)
                curvature = 0
            else:
                #dir = np.sign(r)

                # Find coordinates for center of curved segment
                x_c = x_prev - r*(np.sin(psi_prev))
                y_c = y_prev + r*(np.cos(psi_prev))
                # Angle spanned by curved segment
                theta = l/r
                #end of curve
                x = x_c + r*np.sin(psi_prev + theta)
                y = y_c - r*np.cos(psi_prev + theta)
                #curvature of segment
                curvature = 1/r

                #next angle
                psi = wrap_angle(psi_prev + theta)
            cum_s = cum_s_prev + l
            track_key_pts[i] = np.array([x, y, psi, cum_s, l, curvature])

        return track_key_pts

    def plot_map(self, ax, pts_per_dist=None, close_loop=True):
        if self.key_pts is None:
            raise ValueError('Track key points have not been defined')

        if pts_per_dist is None:
            pts_per_dist = 2000 / self.track_length

        # Plot the starting line
        init_x = self.key_pts[0,0]
        init_y = self.key_pts[0,1]
        init_psi = self.key_pts[0,2]
        start_line_x = [init_x + np.cos(init_psi+np.pi/2)*self.track_width/2,
                        init_x - np.cos(init_psi+np.pi/2)*self.track_width/2]
        start_line_y = [init_y + np.sin(init_psi+np.pi/2)*self.track_width/2,
                        init_y - np.sin(init_psi+np.pi/2)*self.track_width/2]
        ax.plot(start_line_x, start_line_y, 'r', linewidth=1)

        # Plot the track and boundaries
        x_track = []
        x_bound_in = []
        x_bound_out = []
        y_track = []
        y_bound_in = []
        y_bound_out = []
        for i in range(1, self.key_pts.shape[0]):
            l = self.key_pts[i,4]
            cum_s = self.key_pts[i-1,3]
            n_pts = np.around(l*pts_per_dist)
            d = np.linspace(0, l, int(n_pts))  #FIXED - numpy no longer allows interpolation using a double number of points
            for j in d:
                cl_coord = (j + cum_s, 0, 0)
                xy_coord = self.local_to_global(cl_coord)
                x_track.append(xy_coord[0])
                y_track.append(xy_coord[1])
                cl_coord = (j + cum_s, self.track_width/2, 0)
                xy_coord = self.local_to_global(cl_coord)
                x_bound_in.append(xy_coord[0])
                y_bound_in.append(xy_coord[1])
                cl_coord = (j + cum_s, -self.track_width/2, 0)
                xy_coord = self.local_to_global(cl_coord)
                x_bound_out.append(xy_coord[0])
                y_bound_out.append(xy_coord[1])
        if close_loop:
            ax.plot(x_track, y_track, 'k--', linewidth=1)
            ax.plot(x_bound_in, y_bound_in, 'k')
            ax.plot(x_bound_out, y_bound_out, 'k')
        else:
            ax.plot(x_track[:-1], y_track[:-1], 'k--', linewidth=1)
            ax.plot(x_bound_in[:-1], y_bound_in[:-1], 'k')
            ax.plot(x_bound_out[:-1], y_bound_out[:-1], 'k')

        track_bbox = (np.amin(x_bound_out),
                        np.amin(y_bound_out),
                        np.amax(x_bound_out)-np.amin(x_bound_out),
                        np.amax(y_bound_out)-np.amin(y_bound_out))
        
        return track_bbox

    def plot_map_qt(self, p, pts_per_dist=None, close_loop=True):
        if self.key_pts is None:
            raise ValueError('Track key points have not been defined')

        if pts_per_dist is None:
            pts_per_dist = 2000 / self.track_length

        # Plot the starting line
        init_x = self.key_pts[0,0]
        init_y = self.key_pts[0,1]
        init_psi = self.key_pts[0,2]
        start_line_x = [init_x + np.cos(init_psi+np.pi/2)*self.track_width/2,
                        init_x - np.cos(init_psi+np.pi/2)*self.track_width/2]
        start_line_y = [init_y + np.sin(init_psi+np.pi/2)*self.track_width/2,
                        init_y - np.sin(init_psi+np.pi/2)*self.track_width/2]

        p.plot(start_line_x, start_line_y, pen=pg.mkPen('r', width=1))

        # Plot the track and boundaries
        x_track = []
        x_bound_in = []
        x_bound_out = []
        y_track = []
        y_bound_in = []
        y_bound_out = []
        for i in range(1, self.key_pts.shape[0]):
            l = self.key_pts[i,4]
            cum_s = self.key_pts[i-1,3]
            n_pts = np.around(l*pts_per_dist)
            d = np.linspace(0, l, int(n_pts))  #FIXED - numpy no longer allows interpolation using a double number of points
            for j in d:
                cl_coord = (j + cum_s, 0, 0)
                xy_coord = self.local_to_global(cl_coord)
                x_track.append(xy_coord[0])
                y_track.append(xy_coord[1])
                cl_coord = (j + cum_s, self.track_width/2, 0)
                xy_coord = self.local_to_global(cl_coord)
                x_bound_in.append(xy_coord[0])
                y_bound_in.append(xy_coord[1])
                cl_coord = (j + cum_s, -self.track_width/2, 0)
                xy_coord = self.local_to_global(cl_coord)
                x_bound_out.append(xy_coord[0])
                y_bound_out.append(xy_coord[1])
        if close_loop:
            p.plot(x_track, y_track, pen=pg.mkPen('k', width=1, dash=[4,2]))
            p.plot(x_bound_in, y_bound_in, pen=pg.mkPen('k', width=2))
            p.plot(x_bound_out, y_bound_out, pen=pg.mkPen('k', width=2))
        else:
            p.plot(x_track[:-1], y_track[:-1], pen=pg.mkPen('k', width=1, dash=[4,2]))
            p.plot(x_bound_in[:-1], y_bound_in[:-1], pen=pg.mkPen('k', width=2))
            p.plot(x_bound_out[:-1], y_bound_out[:-1], pen=pg.mkPen('k', width=2))

        track_bbox = (np.amin(x_bound_out),
                        np.amin(y_bound_out),
                        np.amax(x_bound_out)-np.amin(x_bound_out),
                        np.amax(y_bound_out)-np.amin(y_bound_out))

        return track_bbox

    def remove_phase_out(self):
        if self.phase_out:
            self.track_length = self.key_pts[-2][3]
            self.key_pts = self.key_pts[0:-1]
            self.n_segs = self.n_segs - 1
            self.cl_segs = self.cl_segs[0:-1]
            self.phase_out = False

    def global_to_local_2(self, xy_coord):
        if self.key_pts is None:
            raise ValueError('Track key points have not been defined')

        x, y, psi = xy_coord

        pos_cur = np.array([x, y])

        d2seg = np.zeros(self.key_pts.shape[0]-1)
        for i in range(1, self.key_pts.shape[0]):
            x_s = self.key_pts[i-1,0]
            y_s = self.key_pts[i-1,1]
            psi_s = self.key_pts[i-1,2]
            x_f = self.key_pts[i,0]
            y_f = self.key_pts[i,1]
            curve_f = self.key_pts[i,5]

            l = self.key_pts[i,4]

            pos_s = np.array([x_s, y_s])
            pos_f = np.array([x_f, y_f])

            e_y = np.inf
            if curve_f == 0:
                if np.abs(compute_angle(pos_s, pos_cur, pos_f)) <= np.pi/2 and np.abs(compute_angle(pos_f, pos_cur, pos_s)) <= np.pi/2:
                    # Check if on straight segment
                    v = pos_cur - pos_s
                    ang = compute_angle(pos_s, pos_f, pos_cur)
                    e_y = la.norm(v)*np.sin(ang)
            else:
                # Check if on curved segment
                r = 1/curve_f
                dir = np.sign(r)

                # Find coordinates for center of curved segment
                x_c = x_s + np.abs(r)*np.cos(psi_s + dir*np.pi/2)
                y_c = y_s + np.abs(r)*np.sin(psi_s + dir*np.pi/2)
                curve_center = np.array([x_c, y_c])

                span_ang = l/r
                cur_ang = compute_angle(curve_center, pos_s, pos_cur)
                if np.sign(span_ang) == np.sign(cur_ang) and np.abs(span_ang) >= np.abs(cur_ang):
                    v = pos_cur - curve_center
                    e_y = -np.sign(dir)*(la.norm(v)-np.abs(r))
            d2seg[i-1] = e_y

        seg_idx = np.argmin(np.abs(d2seg)) + 1

        x_s = self.key_pts[seg_idx-1,0]
        y_s = self.key_pts[seg_idx-1,1]
        psi_s = self.key_pts[seg_idx-1,2]
        s_s = self.key_pts[seg_idx-1,3]
        x_f = self.key_pts[seg_idx,0]
        y_f = self.key_pts[seg_idx,1]
        curve_f = self.key_pts[seg_idx,5]

        pos_s = np.array([x_s, y_s])
        pos_f = np.array([x_f, y_f])

        if curve_f == 0:
            # Check if on straight segment
            v = pos_cur - pos_s
            ang = compute_angle(pos_s, pos_f, pos_cur)
            d = la.norm(v)*np.cos(ang)
        else:
            # Check if on curved segment
            r = 1/curve_f
            dir = np.sign(r)

            # Find coordinates for center of curved segment
            x_c = x_s + np.abs(r)*np.cos(psi_s + dir*np.pi/2)
            y_c = y_s + np.abs(r)*np.sin(psi_s + dir*np.pi/2)
            curve_center = np.array([x_c, y_c])
            cur_ang = compute_angle(curve_center, pos_s, pos_cur)
            d = np.abs(cur_ang)*np.abs(r)

        return (s_s + d, d2seg[seg_idx-1], 0)

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

        x = xy_coord[0]
        y = xy_coord[1]
        psi = xy_coord[2]

        pos_cur = np.array([x, y])
        cl_coord = None

        for i in range(1, self.key_pts.shape[0]):
            # key_pts = [x, y, psi, cumulative length, segment length, signed curvature]
            x_s = self.key_pts[i-1,0]
            y_s = self.key_pts[i-1,1]
            psi_s = self.key_pts[i-1,2]
            curve_s = self.key_pts[i-1,5]
            x_f = self.key_pts[i,0]
            y_f = self.key_pts[i,1]
            psi_f = self.key_pts[i,2]
            curve_f = self.key_pts[i,5]

            l = self.key_pts[i,4]

            pos_s = np.array([x_s, y_s])
            pos_f = np.array([x_f, y_f])

            # Check if at any of the segment start or end points
            if la.norm(pos_s - pos_cur) == 0:
                # At start of segment
                s = self.key_pts[i-1,3]
                e_y = 0
                e_psi = np.unwrap([psi_s, psi])[1] - psi_s
                cl_coord = (s, e_y, e_psi)
                break
            if la.norm(pos_f - pos_cur) == 0:
                # At end of segment
                s = self.key_pts[i,3]
                e_y = 0
                e_psi = np.unwrap([psi_f, psi])[1] - psi_f
                cl_coord = (s, e_y, e_psi)
                break

            if curve_f == 0:
                # Check if on straight segment
                if np.abs(compute_angle(pos_s, pos_cur, pos_f)) <= np.pi/2 and np.abs(compute_angle(pos_f, pos_cur, pos_s)) <= np.pi/2:
                    v = pos_cur - pos_s
                    ang = compute_angle(pos_s, pos_f, pos_cur)
                    e_y = la.norm(v)*np.sin(ang)
                    # Check if deviation from centerline is within track width plus some slack for current segment
                    # (allows for points outside of track boundaries)
                    if np.abs(e_y) <= self.track_width/2 + self.slack:
                        d = la.norm(v)*np.cos(ang)
                        s = self.key_pts[i-1,3] + d
                        e_psi = np.unwrap([psi_s, psi])[1] - psi_s
                        cl_coord = (s, e_y, e_psi)
                        break
                    else:
                        continue
                else:
                    continue
            else:
                # Check if on curved segment
                r = 1/curve_f
                dir = np.sign(r)

                # Find coordinates for center of curved segment
                x_c = x_s + np.abs(r)*np.cos(psi_s + dir*np.pi/2)
                y_c = y_s + np.abs(r)*np.sin(psi_s + dir*np.pi/2)
                curve_center = np.array([x_c, y_c])

                span_ang = l/r
                cur_ang = compute_angle(curve_center, pos_s, pos_cur)
                if np.sign(span_ang) == np.sign(cur_ang) and np.abs(span_ang) >= np.abs(cur_ang):
                    v = pos_cur - curve_center
                    e_y = -np.sign(dir)*(la.norm(v)-np.abs(r))
                    # Check if deviation from centerline is within track width plus some slack for current segment
                    # (allows for points outside of track boundaries)
                    if np.abs(e_y) <= self.track_width/2 + self.slack:
                        d = np.abs(cur_ang)*np.abs(r)
                        s = self.key_pts[i-1,3] + d
                        e_psi = np.unwrap([psi_s+cur_ang, psi])[1] - (psi_s+cur_ang)
                        cl_coord = (s, e_y, e_psi)
                        break
                    else:
                        continue
                else:
                    continue

        if line == 'inside':
            cl_coord = (cl_coord[0], cl_coord[1]-self.track_width/5, cl_coord[2])
        elif line == 'outside':
            cl_coord = (cl_coord[0], cl_coord[1]+self.track_width/5, cl_coord[2])
        elif line == 'pid_offset':
            # PID controller tends to cut to the inside of the track
            cl_coord = (cl_coord[0], cl_coord[1]+(0.1*self.track_width/2), cl_coord[2])

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

        #s = np.mod(cl_coord[0], self.track_length) # Distance along current lap
        s = cl_coord[0]
        while s < 0: s += self.track_length
        while s >= self.track_length: s -= self.track_length

        e_y = cl_coord[1]
        e_psi = cl_coord[2]

        # Find key point indicies corresponding to current segment
        # key_pts = [x, y, psi, cumulative length, segment length, signed curvature]
        key_pt_idx_s = np.where(s >= self.key_pts[:,3])[0][-1]
        key_pt_idx_f = key_pt_idx_s + 1
        seg_idx = key_pt_idx_s

        x_s = self.key_pts[key_pt_idx_s,0]
        y_s = self.key_pts[key_pt_idx_s,1]
        psi_s = self.key_pts[key_pt_idx_s,2]
        curve_s = self.key_pts[key_pt_idx_s,5]
        x_f = self.key_pts[key_pt_idx_f,0]
        y_f = self.key_pts[key_pt_idx_f,1]
        psi_f = self.key_pts[key_pt_idx_f,2]
        curve_f = self.key_pts[key_pt_idx_f,5]

        l = self.key_pts[key_pt_idx_f,4]
        d = s - self.key_pts[key_pt_idx_s,3] # Distance along current segment

        if curve_f == 0:
            # Segment is a straight line
            x = x_s + (x_f - x_s)*d/l + e_y*np.cos(psi_f + np.pi/2)
            y = y_s + (y_f - y_s)*d/l + e_y*np.sin(psi_f + np.pi/2)
            psi = wrap_angle(psi_f + e_psi)
        else:
            r = 1/curve_f
            dir = sign(r)

            # Find coordinates for center of curved segment
            x_c = x_s + np.abs(r)*np.cos(psi_s + dir*np.pi/2)
            y_c = y_s + np.abs(r)*np.sin(psi_s + dir*np.pi/2)

            # Angle spanned up to current location along segment
            span_ang = d/np.abs(r)

            # Angle of the tangent vector at the current location
            psi_d = wrap_angle(psi_s + dir*span_ang)

            ang_norm = wrap_angle(psi_s + dir*np.pi/2)
            ang = -sign(ang_norm)*(np.pi-np.abs(ang_norm))

            x = x_c + (np.abs(r) - dir*e_y)*np.cos(ang + dir*span_ang)
            y = y_c + (np.abs(r) - dir*e_y)*np.sin(ang + dir*span_ang)
            psi = wrap_angle(psi_d + e_psi)
        return (x, y, psi)


    def generate_texture(self,n_s = 1000, n_w = 100,thickness = None):
        if thickness is None:
            thickness = self.track_width / 10

        # grid the track as a parametric surface
        s_array = np.linspace(0,self.track_length, n_s)
        w_array = np.linspace(-self.track_width/2, self.track_width/2, n_w)

        S_array, W_array = np.meshgrid(s_array, w_array)

        V_array = np.zeros((S_array.size, 3))

        # compute x,y,z as a function of every grid point
        for i in range(n_s):
            for j in range(n_w):
                x,y,_ = self.local_to_global((s_array[i], w_array[j], 0))
                #z = 0.2*np.sin(s_array[i] / self.track_length * 10*np.pi)
                z = 0
                V_array[i*n_w + j] = np.array([x,y,z])

        # now compute indices that correspond to triangles
        I_array = np.zeros((2*(n_s-1) * (n_w)-1, 3))
        for i in range(n_s-1):
            for j in range(n_w-1):
                #lower and upper triangle
                l = np.array([i*n_w + j,    (i+1)*n_w + j,  i*n_w + j + 1])
                u = np.array([(i+1)*n_w + j, i*n_w + j + 1, (i+1)*n_w + j + 1])

                I_array[2*(i*n_w + j)] = l
                I_array[2*(i*n_w + j) + 1] = u

        # This is enough to get a 2D surface, now lets offset it by thickness in the z direction and get a "solid" body
        #first generate an offset surface
        V_array_lower = V_array.copy()
        V_array_lower[:,2] -= thickness
        I_array_lower = I_array.copy()

        #combine the surfaces
        V_tot = np.vstack([V_array, V_array_lower])
        offset = V_array.shape[0]
        I_array_lower += offset
        I_tot = np.vstack([I_array, I_array_lower])

        # add triangles along the edges
        #pdb.set_trace()

        # s sides:
        for i in range(n_s-1):
            #left lower and upper triangles
            ll = np.array([i*n_w, (i+1)*n_w, i*n_w + offset])
            lu = np.array([(i+1)*n_w, (i+1)*n_w+ offset, i*n_w + offset])

            #right lower and upper triangles
            rl = np.array([i*n_w + n_w-1,   (i+1)*n_w+ n_w-1,        i*n_w + offset + n_w-1])
            ru = np.array([(i+1)*n_w+ n_w-1,(i+1)*n_w+ offset+ n_w-1, i*n_w + offset+ n_w-1])

            I_tot = np.vstack([I_tot, ll, lu, rl, ru])

        # w sides:
        for j in range(n_w-1):
            #left lower and upper triangles
            ll = np.array([j, j+1, j + offset])
            lu = np.array([j+1, j+1+ offset, j+ offset])

            #right lower and upper triangles
            ll = np.array([j + n_s-1, j+1 + n_s-1, j + offset + n_s-1])
            lu = np.array([j+1 + n_s-1, j+1+ offset + n_s-1, j+ offset + n_s-1])

            I_tot = np.vstack([I_tot, ll, lu, rl, ru])

        return V_tot, I_tot


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
    det = v_1[0]*v_2[1] - v_1[1]*v_2[0]
    theta = np.arctan2(det, dot)

    return theta

def plot_tests():
    from mpclab_common.track import get_track
    from pyqtgraph.Qt import QtGui, QtCore

    track = get_track('Lab_Track_barc')

    print(f"Track Width: {track.track_width}")
    print(f"Track Length (spline): {track.track_length}")

    # matplotlib
    # fig = plt.figure()
    # ax = plt.subplot(111)
    # track.plot_map(ax)
    # track.plot_approximations(ax)
    # plt.show()

    # QT
    app = QtGui.QApplication([])
    pg.setConfigOptions(antialias=True, background='w')
    figsize = (750, 750)
    widget = pg.GraphicsLayoutWidget(show=True)
    widget.setWindowTitle('BARC Plotter')
    widget.resize(*figsize)
    l_xy = widget.addLayout()
    vb_xy = l_xy.addViewBox(lockAspect=True)
    p_xy = l_xy.addPlot(viewBox=vb_xy)
    p_xy.setDownsampling(mode='peak')
    p_xy.setLabel('left', 'Y', units='m')
    p_xy.setLabel('bottom', 'X', units='m')
    track_bbox = track.plot_map_qt(p_xy)  # plot the track once at the start and avoid deleting it.
    print(track_bbox)
    app.exec_()


if __name__ == "__main__":
    plot_tests()
