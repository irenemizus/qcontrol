import math_base


class GridConstructor:
    def __init__(self, conf_prop):
        self.L = conf_prop.L
        self.np = conf_prop.np
    def grid_setup(self):
        # calculating coordinate step of the problem
        dx = self.L / self.np

        # setting the coordinate grid
        x = math_base.coord_grid(dx, self.np)

        return dx, x


class ForwardTimeGridConstructor:
    def __init__(self, conf_prop):
        self.T = conf_prop.T
        self.nt = conf_prop.nt
    def grid_setup(self):
        # calculating time step of the problem
        dt = self.T / self.nt

        # setting the time grid
        t = []
        for l in range(self.nt + 1):
            t.append(dt * l)

        return dt, t
