import math_base


class GridConstructor:
    def __init__(self, conf_prop):
        self.L = conf_prop.L
        self.np = conf_prop.np
    def grid_setup(self):
        # calculating coordinate step of the problem
        dx = self.L / (self.np - 1)

        # setting the coordinate grid
        x = math_base.coord_grid(dx, self.np)

        return dx, x