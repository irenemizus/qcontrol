import numpy


class GridConstructor:
    def __init__(self, conf_task):
        self.L = conf_task.L
        self.np = conf_task.np
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
        if self.np > 1:
            dx: numpy.float64
            dx = self.L / (self.np - 1)

            # setting the coordinate grid
            shift = numpy.float64(self.np - 1) * dx / 2.0
            x_list = [numpy.float64(i) * dx - shift for i in range(self.np)]
            x = numpy.array(x_list)
        elif self.np == 1:
            dx = numpy.float64(1.0)
            x = numpy.array([ 0.0 ])
        else:
            raise RuntimeError("The number of collocation points 'np' must be positive integers!")

        return dx, x


class ForwardTimeGridConstructor:
    def __init__(self, conf_task):
        self.T = conf_task.T
        self.nt = conf_task.fitter.propagation.nt
    def grid_setup(self):
        # calculating time step of the problem
        dt = self.T / self.nt

        # setting the time grid
        t = []
        for l in range(self.nt + 1):
            t.append(dt * l)

        return dt, t
