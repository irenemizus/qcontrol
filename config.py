import copy
import typing
from enum import Enum
from typing import Dict


class ConfigurationBase:
    _data: Dict[str, typing.Any]

    def __init__(self, key_prefix: str):
        self._empty = True
        self._key_prefix = key_prefix
        self._data = {
        }

    def is_empty(self):
        return self._empty

    def load(self, user_data):
        self._empty = False  # Even loading an empty configuration means the object isn't empty anymore
        # analyze provided input json user_data
        for key in self._data:
            key_with_prefix = self._key_prefix + key
            actual_key = ""
            if key_with_prefix in user_data:
                actual_key = key_with_prefix
            elif key in user_data:
                actual_key = key

            if actual_key != "":
                if isinstance(self._data[key], str):
                    self._data[key] = str(user_data[actual_key])
                elif isinstance(self._data[key], float):
                    self._data[key] = float(user_data[actual_key])
                elif isinstance(self._data[key], int):
                    self._data[key] = int(user_data[actual_key])
                elif isinstance(user_data[actual_key], str):
                    self._data[key] = type(self._data[key]).from_str(str(user_data[actual_key]))
                elif isinstance(user_data[actual_key], int):
                    self._data[key] = type(self._data[key]).from_int(int(user_data[actual_key]))
                elif isinstance(user_data[actual_key], float):
                    self._data[key] = type(self._data[key]).from_float(float(user_data[actual_key]))
                elif isinstance(self._data[key], ConfigurationBase):
                    # Careful! Recursion here
                    assert(self._data[key]._key_prefix == self._key_prefix)
                    self._data[key].load(user_data[actual_key])
            else:
                print(
                    "Parameter '%s' hasn't been provided in the input json file. "
                    "The default value will be used: %s" % (key, str(self._data[key]))
                )

    # redefinition of the dot operator for a field
    def __getattr__(self, key):
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(key)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

##########################################

class TaskRootConfiguration(ConfigurationBase):
    def __init__(self):
        super().__init__(key_prefix="")
        self._data["fitter"] = TaskRootConfiguration.FitterConfiguration()

    class FitterConfiguration(ConfigurationBase):
        class PropagationConfiguration(ConfigurationBase):
            class WaveFuncType(Enum):
                MORSE = 0
                HARMONIC = 1
                CONST = 2

                @staticmethod
                def from_int(i):
                    return TaskRootConfiguration.FitterConfiguration.\
                        PropagationConfiguration.WaveFuncType(i)

                @staticmethod
                def from_str(s):
                    return TaskRootConfiguration.FitterConfiguration.\
                        PropagationConfiguration.WaveFuncType[s.upper()]


            class PotentialType(Enum):
                MORSE = 0
                HARMONIC = 1
                NONE = 2

                @staticmethod
                def from_int(i):
                    return TaskRootConfiguration.FitterConfiguration.\
                        PropagationConfiguration.PotentialType(i)

                @staticmethod
                def from_str(s):
                    return TaskRootConfiguration.FitterConfiguration.\
                        PropagationConfiguration.PotentialType[s.upper()]

            def __init__(self):
                super().__init__(key_prefix="")
                # default input values
                self._data["m"] = 0.5   # Dalton
                # 1.0 -- for a model harmonic oscillator
                self._data["wf_type"] = TaskRootConfiguration.FitterConfiguration.PropagationConfiguration.WaveFuncType.MORSE
                self._data["pot_type"] = TaskRootConfiguration.FitterConfiguration.PropagationConfiguration.PotentialType.MORSE
                self._data["hamil_type"] = TaskRootConfiguration.FitterConfiguration.HamilType.NTRIV
                self._data["U"] = 0.0
                self._data["delta"] = 0.0
                self._data["a"] = 1.0   # 1/a_0 -- for morse oscillator, a_0 -- for harmonic oscillator
                self._data["De"] = 20000.0  # 1/cm
                self._data["x0p"] = -0.17   # a_0
                self._data["a_e"] = 1.0
                self._data["De_e"] = 10000.0
                self._data["Du"] = 20000.0
                self._data["x0"] = 0.0  # TODO: to fix x0 != 0
                self._data["p0"] = 0.0  # TODO: to fix p0 != 0
                self._data["L"] = 5.0   # a_0
                # 5.0 a_0 -- for the working transition between PECs and controls;
                # 0.2 -- for a model harmonic oscillator with a = 1.0;
                # 4.0 a_0 -- for morse oscillator;
                # 10.0 a_0 -- for dimensional harmonic oscillator
                self._data["T"] = 600e-15   # s
                # 1200 fs -- for two laser pulses;
                # 280 (600) fs -- for the working transition between PECs and LC;
                # 2240 fs -- for filtering on the ground PEC (99.16% quality)
                # 0.1 pi (half period units) -- for a model harmonic oscillator
                self._data["np"] = 1024
                # 1024 -- for the working transition between PECs and controls;
                # 128 -- for a model harmonic oscillator with a = 1.0;
                # 2048 -- for morse oscillator and filtering on the ground PEC (99.16% quality);
                # 512 -- for dimensional harmonic oscillator
                self._data["nch"] = 64
                self._data["nt"] = 420000
                # 840000 -- for two laser pulses;
                # 200000 (420000) -- for the working transition between PECs and LC;
                # 900000 -- for filtering on the ground PEC (99.16% quality)
                self._data["E0"] = 71.54    # 1/cm
                self._data["t0"] = 300e-15  # s
                self._data["sigma"] = 50e-15    # s
                self._data["nu_L"] = 0.29297e15 # Hz
                # 0.29297e15 -- for the working transition between PECs;
                # 0.5879558e15 -- analytical difference b/w excited and ground energies;
                # 0.5859603e15 -- calculated difference b/w excited and ground energies !!;
                # 0.599586e15 = 20000 1/cm


        class TaskType(Enum):
            SINGLE_POT = 0
            FILTERING = 1
            TRANS_WO_CONTROL = 2
            INTUITIVE_CONTROL = 3
            LOCAL_CONTROL_POPULATION = 4
            LOCAL_CONTROL_PROJECTION = 5
            OPTIMAL_CONTROL_KROTOV = 6
            OPTIMAL_CONTROL_UNIT_TRANSFORM = 7

            @staticmethod
            def from_int(i):
                return TaskRootConfiguration.FitterConfiguration.TaskType(i)

            @staticmethod
            def from_str(s):
                return TaskRootConfiguration.FitterConfiguration.TaskType[s.upper()]


        class InitGuess(Enum):
            ZERO = 0
            GAUSS = 1
            SQRSIN = 2
            MAXWELL = 3

            @staticmethod
            def from_int(i):
                return TaskRootConfiguration.FitterConfiguration.InitGuess(i)

            @staticmethod
            def from_str(s):
                return TaskRootConfiguration.FitterConfiguration.InitGuess[s.upper()]


        class HamilType(Enum):
            NTRIV = 0
            TWO_LEVELS = 1
            ANG_MOMS = 2

            @staticmethod
            def from_int(i):
                return TaskRootConfiguration.FitterConfiguration.HamilType(i)

            @staticmethod
            def from_str(s):
                return TaskRootConfiguration.FitterConfiguration.HamilType[s.upper()]


        class InitGuessHf(Enum):
            EXP = 0
            COS = 1
            COS_SET = 2

            @staticmethod
            def from_int(i):
                return TaskRootConfiguration.FitterConfiguration.InitGuessHf(i)

            @staticmethod
            def from_str(s):
                return TaskRootConfiguration.FitterConfiguration.InitGuessHf[s.upper()]


        def __init__(self):
            super().__init__(key_prefix="")
            # default input values
            self._data["task_type"] = TaskRootConfiguration.FitterConfiguration.TaskType.TRANS_WO_CONTROL
            self._data["init_guess"] = TaskRootConfiguration.FitterConfiguration.InitGuess.ZERO
            self._data["init_guess_hf"] = TaskRootConfiguration.FitterConfiguration.InitGuessHf.EXP
            self._data["propagation"] = TaskRootConfiguration.FitterConfiguration.PropagationConfiguration()
            self._data["k_E"] = 1e29    # 1 / (s*s)
            self._data["lamb"] = 4e14 # 1 / s
            self._data["pow"] = 0.8
            self._data["epsilon"] = 1e-15
            self._data["impulses_number"] = 2
            self._data["delay"] = 600e-15   # s
            self._data["mod_log"] = 500
            self._data["iter_max"] = -1
            self._data["h_lambda"] = 0.0066
            self._data["nb"] = 1
            self._data["pcos"] = 3
            self._data["Em"] = 1.5


class ReportRootConfiguration(ConfigurationBase):
    def __init__(self, key_prefix: str):
        super().__init__(key_prefix)
        self._data["fitter"] = ReportRootConfiguration.ReportFitterConfiguration(key_prefix=key_prefix)

    class ReportFitterConfiguration(ConfigurationBase):
        def __init__(self, key_prefix: str):
            super().__init__(key_prefix)
            self._data["propagation"] = ReportRootConfiguration.ReportFitterConfiguration.ReportPropagationConfiguration(key_prefix=key_prefix)
            self._data["out_path"] = "output"

        class ReportPropagationConfiguration(ConfigurationBase):
            def __init__(self, key_prefix: str):
                super().__init__(key_prefix)

        class ReportTablePropagationConfiguration(ReportPropagationConfiguration):
            def __init__(self, suffix=None):
                if suffix is not None and suffix != "":
                    suffix = "_" + suffix
                else:
                    suffix = ""

                super().__init__(key_prefix="table.")
                # default input values
                self._data["tab_abs"] = "tab_abs_{level}.csv"
                self._data["tab_real"] = "tab_real_{level}.csv"
                self._data["tab_tvals"] = "tab_tvals_{level}.csv"
                self._data["tab_tvals_fit"] = "tab_tvals_fit.csv"
                self._data["lmin"] = 0
                self._data["mod_fileout"] = 100

        class ReportPlotPropagationConfiguration(ReportPropagationConfiguration):
            def __init__(self, suffix=None):
                super().__init__(key_prefix="plot.")
                if suffix is not None and suffix != "":
                    suffix = "_" + suffix
                else:
                    suffix = ""

                # default input values
                self._data["lmin"] = 0
                self._data["mod_plotout"] = 100
                self._data["mod_update"] = 20
                self._data["number_plotout"] = 15

                self._data["gr_abs"] = "fig_abs{level}.pdf"
                self._data["gr_real"] = "fig_real{level}.pdf"
                self._data["gr_moms_low"] = "fig_moms_low{level}.pdf"
                self._data["gr_moms"] = "fig_moms{level}.pdf"

                self._data["gr_ener"] = f"fig_ener{suffix}.pdf"
                self._data["gr_overlp0"] = f"fig_overlp0{suffix}.pdf"
                self._data["gr_overlpf"] = f"fig_overlpf{suffix}.pdf"
                self._data["gr_abs_max"] = f"fig_abs_max{suffix}.pdf"
                self._data["gr_real_max"] = f"fig_real_max{suffix}.pdf"

                self._data["gr_ener_tot"] = f"fig_ener_tot{suffix}.pdf"
                self._data["gr_overlp0_tot"] = f"fig_overlp0_tot{suffix}.pdf"
                self._data["gr_overlpf_tot"] = f"fig_overlpf_tot{suffix}.pdf"
                self._data["gr_lf_en"] = f"fig_lf_en{suffix}.pdf"
                self._data["gr_lf_fr"] = f"fig_lf_fr{suffix}.pdf"

    class ReportTableFitterConfiguration(ReportFitterConfiguration):
        def __init__(self, suffix=None):
            super().__init__(key_prefix="table.")
            self._data["propagation"] = ReportRootConfiguration.ReportFitterConfiguration.ReportTablePropagationConfiguration()

            # default input values
            self._data["tab_iter"] = "tab_iter.csv"
            self._data["tab_iter_E"] = "tab_iter_E.csv"
            self._data["imin"] = 0
            self._data["imod_fileout"] = 1

    class ReportPlotFitterConfiguration(ReportFitterConfiguration):
        def __init__(self, suffix=None):
            super().__init__(key_prefix="plot.")
            self._data["propagation"] = ReportRootConfiguration.ReportFitterConfiguration.ReportPlotPropagationConfiguration()

            # default input values
            self._data["imin"] = 0
            self._data["imod_plotout"] = 1
            self._data["inumber_plotout"] = 15
            self._data["gr_iter"] = "fig_iter.pdf"
            self._data["gr_iter_E"] = "fig_iter_E.pdf"

class ReportTableRootConfiguration(ReportRootConfiguration):
    def __init__(self):
        super().__init__(key_prefix="table.")
        self._data["fitter"] = ReportRootConfiguration.ReportTableFitterConfiguration()

class ReportPlotRootConfiguration(ReportRootConfiguration):
    def __init__(self):
        super().__init__(key_prefix="plot.")
        self._data["fitter"] = ReportRootConfiguration.ReportPlotFitterConfiguration()
