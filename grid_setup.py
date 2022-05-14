import numpy


class GridConstructor:
    def __init__(self, conf_prop):
        self.L = conf_prop.L
        self.np = conf_prop.np

    def grid_setup(self):
        """ Setting of the coordinate grid; it should be symmetric,
            equidistant and centered at about minimum of the potential
            INPUT
            L   spatial range of the problem
            np  number of grid points
            OUTPUT
            dx coordinate grid step
            x  vector of length np defining positions of grid points """

        # calculating coordinate step of the problem
        dx = self.L / self.np

        # setting the coordinate grid
        shift = float(self.np - 1) * dx / 2.0
        x_list = [float(i) * dx - shift for i in range(self.np)]
        x = numpy.array(x_list)
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
