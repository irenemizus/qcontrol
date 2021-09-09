from enum import Enum

class ConfigurationBase:
    def __init__(self):
        self._data = {
        }

    def load(self, user_data):
        # analyze provided input json user_data
        for key in self._data:
            if key in user_data:
                if isinstance(self._data[key], str):
                    self._data[key] = str(user_data[key])
                elif isinstance(self._data[key], float):
                    self._data[key] = float(user_data[key])
                elif isinstance(self._data[key], int):
                    self._data[key] = int(user_data[key])
                elif isinstance(user_data[key], str):
                    self._data[key] = type(self._data[key]).from_str(str(user_data[key]))
                elif isinstance(user_data[key], int):
                    self._data[key] = type(self._data[key]).from_int(int(user_data[key]))
                elif isinstance(user_data[key], float):
                    self._data[key] = type(self._data[key]).from_float(float(user_data[key]))
                elif isinstance(self._data[key], ConfigurationBase):
                    # Careful! Recursion here
                    self._data[key].load(user_data[key])
            else:
                print(
                    "Parameter '%s' hasn't been provided in the input json file. "
                    "The default value will be used: %s" % (key, str(self._data[key]))
                )

    # redefinition of the dot operator for a field
    def __getattr__(self, key):
        return self._data[key]

##########################################

class RootConfiguration(ConfigurationBase):

    class OutputConfiguration(ConfigurationBase):
        def __init__(self):
            super().__init__()
            # default input values
            self._data["file_abs"] = "fort.21"
            self._data["file_real"] = "fort.22"
            self._data["file_mom"] = "fort.23"
            self._data["lmin"] = 0
            self._data["mod_stdout"] = 500
            self._data["mod_fileout"] = 100

    class FitterConfiguration(ConfigurationBase):
        class PropagationConfiguration(ConfigurationBase):
            class WaveFuncType(Enum):
                MORSE = 0
                HARMONIC = 1

                @staticmethod
                def from_int(i):
                    return RootConfiguration.FitterConfiguration.\
                        PropagationConfiguration.WaveFuncType(i)

                @staticmethod
                def from_str(s):
                    return RootConfiguration.FitterConfiguration.\
                        PropagationConfiguration.WaveFuncType[s.upper()]


            class PotentialType(Enum):
                MORSE = 0
                HARMONIC = 1

                @staticmethod
                def from_int(i):
                    return RootConfiguration.FitterConfiguration.\
                        PropagationConfiguration.PotentialType(i)

                @staticmethod
                def from_str(s):
                    return RootConfiguration.FitterConfiguration.\
                        PropagationConfiguration.PotentialType[s.upper()]


            def __init__(self):
                super().__init__()
                # default input values
                self._data["m"] = 0.5   # Dalton
                self._data["wf_type"] = RootConfiguration.FitterConfiguration.PropagationConfiguration.WaveFuncType.MORSE
                self._data["pot_type"] = RootConfiguration.FitterConfiguration.PropagationConfiguration.PotentialType.MORSE
                self._data["a"] = 1.0   # 1/a_0 -- for morse oscillator, a_0 -- for harmonic oscillator
                self._data["De"] = 20000.0  # 1/cm
                self._data["x0p"] = -0.17   # a_0
                self._data["x0"] = 0.0  # TODO: to fix x0 != 0
                self._data["p0"] = 0.0  # TODO: to fix p0 != 0
                self._data["L"] = 5.0   # a_0
                # 5.0 a_0 -- for the working transition between PECs and controls;
                # 0.2 -- for a model harmonic oscillator with a = 1.0;
                # 4.0 a_0 -- for morse oscillator;
                # 6.0 a_0 -- for dimensional harmonic oscillator
                self._data["T"] = 600e-15   # s
                # 1200 fs -- for two laser pulses;
                # 280 (600) fs -- for the working transition between PECs and LC;
                # 2240 fs -- for filtering on the ground PEC (99.16% quality)
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
            TRANS_WO_CONTROL = 0
            FILTERING = 1
            INTUITIVE_CONTROL = 2
            LOCAL_CONTROL = 3

            @staticmethod
            def from_int(i):
                return RootConfiguration.FitterConfiguration.TaskType(i)

            @staticmethod
            def from_str(s):
                return RootConfiguration.FitterConfiguration.TaskType[s.upper()]


        class TaskSubType(Enum):
            GOAL_POPULATION = 0
            GOAL_MOMENTUM = 1

            @staticmethod
            def from_int(i):
                return RootConfiguration.FitterConfiguration.TaskSubType(i)

            @staticmethod
            def from_str(s):
                return RootConfiguration.FitterConfiguration.TaskSubType[s.upper()]


        def __init__(self):
            super().__init__()
            # default input values
            self._data["task_type"] = RootConfiguration.FitterConfiguration.TaskType.TRANS_WO_CONTROL
            self._data["task_subtype"] = RootConfiguration.FitterConfiguration.TaskSubType.GOAL_POPULATION
            self._data["propagation"] = RootConfiguration.FitterConfiguration.PropagationConfiguration()
            self._data["k_E"] = 1e29    # 1 / (s*s)
            self._data["lamb"] = 4e14 # 1 / s
            self._data["pow"] = 0.8
            self._data["epsilon"] = 1e-15
            self._data["impulses_number"] = 2
            self._data["delay"] = 600e-15   # s


    def __init__(self):
        super().__init__()
        self._data["output"] = RootConfiguration.OutputConfiguration()
        self._data["fitter"] = RootConfiguration.FitterConfiguration()
