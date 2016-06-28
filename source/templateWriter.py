### The file you want to modify using this class can include both templates and variables
# template_variabletobereplaced:templatefile_vartoinject:value_....>
# var_varname
###


templates = {"inputprototxt": "base_network/my_network/base_files/googlenetbase.prototxt",
             "original":"base_network/my_network/base_files/googlenetbase.prototxt"}


class PrototxtCosetes(object):
    def __init__(self, prototxt_template_file):
        self.template = ''.join(open(templates[prototxt_template_file], 'r').readlines())
        inds_to_be_replaced = self.locate_replaceable_fields(self.template)
        self.replaceable = {}
        for ini, end in inds_to_be_replaced:
            v = self.template[ini:end].split("_")
            if v[0] == 'template':
                template_name = v[1]
                args = {}
                if len(v) >= 2:
                    for i in range(2,len(v)):
                        args[v[i].split(":")[0]] = v[i].split(":")[1]
                temp = PrototxtCosetes(template_name)
                val = temp.getString(args)
                self.template = self.template.replace('<<'+self.template[ini:end]+'>>', val)


    def locate_replaceable_fields(self, st):
        found = []
        now = 0
        while 1:
            res = self._find_between_(st, "<<", ">>", now)
            if res:
                found.append(res)
                now = res[1]
            else:
                return found
        return found



    def _find_between_(self, s, first, last, ini):
        s = s[ini:]
        try:
            start = s.index( first ) + len( first )
            end = s.index( last, start )
            return start+ini, end+ini

        except ValueError:
            return None

    def getString(self, fields_to_replace):
        aux = str(self.template)
        print(fields_to_replace)
        for string_to_replace in fields_to_replace:
            aux = aux.replace('<<var_'+string_to_replace+'>>', fields_to_replace[string_to_replace])
        return aux

px = PrototxtCosetes("original")


