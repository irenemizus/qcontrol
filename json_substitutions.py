import copy

from jsonpath2.path import Path


class JsonSubstitutionsIterator:
    @staticmethod
    def replace_obj(root, obj, value):
        if isinstance(root, dict):
            for k in root.keys():
                if root[k] == obj:
                    root[k] = value

                JsonSubstitutionsIterator.replace_obj(root[k], obj, value)

        elif isinstance(root, list):
            for i in range(len(root)):
                if root[i] == obj:
                    root[i] = value

                JsonSubstitutionsIterator.replace_obj(root[i], obj, value)

    @staticmethod
    def replace_subst_id(root, id, value):
        if isinstance(root, dict):
            for k in root.keys():
                if isinstance(root[k], dict) and \
                        "@SUBST:ID" in root[k] and \
                        root[k]["@SUBST:ID"] == id:
                    root[k] = value

                JsonSubstitutionsIterator.replace_subst_id(root[k], id, value)

        elif isinstance(root, list):
            for k in range(len(root)):

                if isinstance(root[k], dict) and \
                        "@SUBST:ID" in root[k] and \
                        root[k]["@SUBST:ID"] == id:
                    root[k] = value

                JsonSubstitutionsIterator.replace_subst_id(root[k], id, value)

    def __init__(self, json_data_src):
        self.__json_data_src = copy.deepcopy(json_data_src)   # The full JSON template
        self.__substs = []                     # The substitutional parts in the template

        self.__subst_lists = []                # The lists of possible values inside the subst. parts
        self.__ik = []                         # The multidimensional iterator among the possible values in the substs

        self.__zero_length_finish_flag = False

        # # Looking for the substitution templates
        jsonpath_expression = Path.parse_str('$..*[?(@["@SUBST:LIST"])]')
        match = jsonpath_expression.match(self.__json_data_src)

        for m in match:
            val = m.current_value
            JsonSubstitutionsIterator.replace_obj(self.__json_data_src, val,
                                                  {"@SUBST:ID": len(self.__substs)}
            )
            self.__substs.append(val)

        for subst in self.__substs:
            if "@SUBST:LIST" in subst:
                self.__subst_lists.append(subst["@SUBST:LIST"])
                self.__ik.append(0)


    def __next__(self):
        N = len(self.__subst_lists)
        instance = copy.deepcopy(self.__json_data_src)  # Copying the template to keep the original safe

        if N == 0:
            # Special case: no substitutions presented in the template
            if not self.__zero_length_finish_flag:
                self.__zero_length_finish_flag = True
                return instance
            else:
                raise StopIteration()
        else:
            if self.__ik[N - 1] == len(self.__subst_lists[N - 1]):
                raise StopIteration()

            # Substitutioning the substitutions
            for k in range(N):
                JsonSubstitutionsIterator.replace_subst_id(instance, k,
                                                      self.__subst_lists[k][self.__ik[k]]
                )

            # Incrementing the ik
            self.__ik[0] += 1
            # Calculating carries
            for k in range(N - 1):
                if self.__ik[k] == len(self.__subst_lists[k]):
                    self.__ik[k] = 0
                    self.__ik[k + 1] += 1

            return instance

class JsonSubstitutions:
    def __init__(self, json_data_src):
        self.__json_data_src = json_data_src

    def __iter__(self):
        return JsonSubstitutionsIterator(self.__json_data_src)
