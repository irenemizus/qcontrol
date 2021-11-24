import math_base
import phys_base


class TaskManager:
    def __init__(self):
        pass

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e):
        raise NotImplementedError()

    def ener_goal(self, psif, v, akx2, np):
        raise NotImplementedError()

    def init_proximity_to_goal(self, psif, psi, dx, np):
        raise NotImplementedError()


class SingleStateTaskManager(TaskManager):
    def __init__(self, psi_goal_imp):
        super().__init__()
        self.psi_goal_imp = psi_goal_imp

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e):
        return self.psi_goal_imp(x, np, x0, p0, m, De, a)

    def ener_goal(self, psif, v, akx2, np):
        return phys_base.hamil(psif[0], v[0][1], akx2, np)

    def init_proximity_to_goal(self, psif, psi, dx, np):
        return math_base.cprod(psif[0], psi[0], dx, np)


class MultipleStateTaskManager(TaskManager):
    def __init__(self, psi_goal_imp):
        super().__init__()
        self.psi_goal_imp = psi_goal_imp

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e):
        return self.psi_goal_imp(x, np, x0p, p0, m, De_e, a_e)

    def ener_goal(self, psif, v, akx2, np):
        return phys_base.hamil(psif[0], v[1][1], akx2, np)

    def init_proximity_to_goal(self, psif, psi, dx, np):
        return math_base.cprod(psif[0], psi[1], dx, np)

