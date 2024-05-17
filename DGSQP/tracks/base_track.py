from abc import abstractmethod
import numpy as np

import pdb

class BaseTrack():
    @abstractmethod
    def global_to_local(self,data):
        raise NotImplementedError('Cannot call base class')

    @abstractmethod
    def local_to_global(self,data):
        raise NotImplementedError('Cannot call base class')

    @abstractmethod
    def get_curvature(self,s):
        raise NotImplementedError('Cannot call base class')

    @abstractmethod
    def get_halfwidth(self,s):
        raise NotImplementedError('Cannot call base class')

    def get_bankangle(self,s):
        raise NotImplementedError('Cannot call base class')

    def global_to_local_typed(self, data):  # data is vehicleState
        xy_coord = (data.x.x, data.x.y, data.e.psi)
        cl_coord = self.global_to_local(xy_coord)
        if cl_coord:
            data.p.s = cl_coord[0]
            data.p.x_tran = cl_coord[1]
            data.p.e_psi = cl_coord[2]
            return 1
        return 0

    def local_to_global_typed(self, data):
        cl_coord = (data.p.s, data.p.x_tran, data.p.e_psi)
        xy_coord = self.local_to_global(cl_coord)
        if xy_coord:
            data.x.x = xy_coord[0]
            data.x.y = xy_coord[1]
            data.e.psi = xy_coord[2]
            return 1
        return 0
    
    def plot_map(self, ax, pts_per_dist=None, close_loop=True, distance_markers=0):
        track = self.get_track_xy(pts_per_dist, close_loop)
        
        x_start = track['start']['x']
        y_start = track['start']['y']
        x_track = track['center']['x']
        y_track = track['center']['y']
        x_bound_in = track['bound_in']['x']
        x_bound_out = track['bound_out']['x']
        y_bound_in = track['bound_in']['y']
        y_bound_out = track['bound_out']['y']
        
        ax.plot(x_track, y_track, 'k--', linewidth=1)
        ax.plot(x_bound_in, y_bound_in, 'k')
        ax.plot(x_bound_out, y_bound_out, 'k')
        ax.plot(x_start, y_start, 'r', linewidth=1)

        if distance_markers > 0:
            if self.track_length >= 1:
                for s in np.arange(distance_markers, self.track_length, distance_markers):
                    p_i = self.local_to_global((s, float(self.left_width(s)), 0))
                    p_o = self.local_to_global((s, -float(self.right_width(s)), 0))
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
    
    def get_track_xy(self, pts_per_dist=None, close_loop=True):
        if pts_per_dist is None:
            pts_per_dist = 2000 / self.track_length
        
        init_x, init_y, init_psi = self.local_to_global((0, 0, 0))
        start_line_x = [init_x + np.cos(init_psi + np.pi / 2) * float(self.left_width(0)),
                        init_x - np.cos(init_psi + np.pi / 2) * float(self.right_width(0))]
        start_line_y = [init_y + np.sin(init_psi + np.pi / 2) * float(self.left_width(0)),
                        init_y - np.sin(init_psi + np.pi / 2) * float(self.right_width(0))]

        # Plot the track and boundaries
        x_track = []
        x_bound_in = []
        x_bound_out = []
        y_track = []
        y_bound_in = []
        y_bound_out = []

        S = np.linspace(0, self.track_length, int(self.track_length*pts_per_dist))
        for s in S:
            cl_coord = (s, 0, 0)
            xy_coord = self.local_to_global(cl_coord)
            x_track.append(xy_coord[0])
            y_track.append(xy_coord[1])
            cl_coord = (s, float(self.left_width(s)), 0)
            xy_coord = self.local_to_global(cl_coord)
            x_bound_in.append(xy_coord[0])
            y_bound_in.append(xy_coord[1])
            cl_coord = (s, -float(self.right_width(s)), 0)
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
    