import copy

import numpy


class Psi:
    def __init__(self, f: list[numpy.ndarray] = None, lvls: int = 2):
        psi = []
        for lvl in range(lvls):
            if f is None:
                psi.append([None])
            else:
                psi.append(f[lvl])
        self._psi = psi

    def get_psi(self) -> list[numpy.ndarray]:
        return self._psi

    def lvls(self):
        return len(self._psi)

    f = property(get_psi, None, None, 'Internal psi functions representation')

    def __deepcopy__(self, memo):
        res = Psi(lvls=self.lvls())
        for i in range(self.lvls()):
            res._psi[i] = self._psi[i].copy()
        return res


class PsiBasis:
    def __init__(self, n: int, lvls: int = 2):
        # self.psis = [[None] * lvls] * n
        self.n = n
        self.lvls = lvls
        self._psis = []
        for el in range(n):
            self._psis.append(Psi(lvls=lvls))

    def __len__(self):
        return len(self._psis)

    def get_psis(self) -> list[Psi]:
        return self._psis

    def __deepcopy__(self, memo):
        res = PsiBasis(self.n, self.lvls)
        for v in range(self.n):
            res.psis[v] = copy.deepcopy(self.psis[v], memo=memo)
        return res

    psis = property(get_psis, None, None, 'Internal psi basis functions representation')
