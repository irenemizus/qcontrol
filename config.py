from enum import Enum


class PotentialType(Enum):
    MORSE = 0
    HARMONIC = 1


class WaveFuncType(Enum):
    MORSE = 0
    HARMONIC = 1


class TaskType(Enum):
    TRANS_WO_CONTROL = 0
    FILTERING = 1
    INTUITIVE_CONTROL = 2
    LOCAL_CONTROL = 3


class DictWrapper(object):
    def __init__(self, d):
        self.d = d
    def __getattr__(self, key):
        return self.d[key]


class InputConfiguration:
    def __init__(self):
        # Default input values
        self.data = {
            "phys_syst_pars": {
                "m": 0.5  # Dalton
            },
            "potential_pars": {
                "pot_type": PotentialType.MORSE,
                "a": 1.0,  # 1/a_0 -- for morse oscillator, a_0 -- for harmonic oscillator
                "De": 20000.0,  # 1/cm
                "x0p": -0.17  # a_0
            },
            "laser_field_pars": {
                "impulses_number": 1,
                "E0": 71.54,  # 1/cm
                "t0": 300e-15,  # s
                "sigma": 50e-15,  # s
                "nu_L": 0.29297e15,  # Hz
                # 0.29297e15 -- for the working transition between PECs; # 0.5879558e15 -- analytical difference b/w excited and ground energies; # 0.5859603e15 -- calculated difference b/w excited and ground energies !!; # 0.599586e15 = 20000 1/cm
                "delay": 600e-15  # s
            },
            "phys_calc_pars": {
                "task_type": TaskType.TRANS_WO_CONTROL,
                "wf_type": WaveFuncType.MORSE,
                "L": 5.0,  # a_0
                # 5.0 a_0 -- for the working transition between PECs; # 0.2 -- for a model harmonic oscillator with a = 1.0; # 4.0 a_0 -- for morse oscillator; # 6.0 a_0 -- for dimensional harmonic oscillator
                "T": 600e-15  # s
                # 1200 fs -- for two laser pulses; # 280 (600) fs -- for the working transition between PECs; # 2240 fs -- for filtering on the ground PEC (99.16% quality)
            },
            "alg_calc_pars": {
                "np": 1024,
                # 1024 -- for the working transition between PECs and two laser pulses; # 128 -- for a model harmonic oscillator with a = 1.0; # 2048 -- for morse oscillator and filtering on the ground PEC (99.16% quality); # 512 -- for dimensional harmonic oscillator
                "nch": 64,
                "nt": 420000,
                # 840000 -- for two laser pulses; 200000 (420000) -- for the working transition between PECs; # 900000 -- for filtering on the ground PEC (99.16% quality)
                "epsilon": 1e-15
            },
            "init_conditions": {
                "x0": 0.0, # TODO: to fix x0 != 0
                "p0": 0.0  # TODO: to fix p0 != 0
            },
            "print_pars": {
                "lmin": 0,
                "mod_stdout": 500,
                "mod_fileout": 100,
                "file_abs": "fort.21",
                "file_real": "fort.22",
                "file_mom": "fort.23"
            }
        }

        self.wrapped_data = None


    def load(self, user_data):
        # analyze provided input json user_data
        for key_sec in self.data:
            for key_par in self.data[key_sec]:
                if key_sec in user_data:
                    if key_par in user_data[key_sec]:
                        if isinstance(self.data[key_sec][key_par], WaveFuncType):
                            self.data[key_sec][key_par] = WaveFuncType[user_data[key_sec][key_par].upper()]
                        elif isinstance(self.data[key_sec][key_par], PotentialType):
                            self.data[key_sec][key_par] = PotentialType[user_data[key_sec][key_par].upper()]
                        elif isinstance(self.data[key_sec][key_par], TaskType):
                            self.data[key_sec][key_par] = TaskType[user_data[key_sec][key_par].upper()]
                        else:

                            self.data[key_sec][key_par] = user_data[key_sec][key_par]
                    else:
                        print(
                            "Parameter '%s' in section '%s' wasn't provided in the input json file. The default value will be used" % (key_par, key_sec)
                        )
                else:
                    print(
                        "Section '%s' wasn't provided in the input json file. The default values of calculation parameters from this section will be used" % key_sec
                    )
        self.wrapped_data = None

    def __refresh_wrapped_data(self):
        if self.wrapped_data is None:
            self.wrapped_data = {}
            for sect in self.data:
                self.wrapped_data[sect] = DictWrapper(self.data[sect])

    def __getattr__(self, key):
        self.__refresh_wrapped_data()
        return self.wrapped_data[key]
