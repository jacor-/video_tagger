### The file you want to modify using this class can include both templates and variables
# template_variabletobereplaced:templatefile_vartoinject:value_....>
# var_varname
###
import pandas as pd

class PrototxtCosetes(object):
    def __init__(self, prototxt_main_file, map_template2file):
        self.template = ''.join(open(prototxt_main_file, 'r').readlines())
        inds_to_be_replaced = self.locate_replaceable_fields(self.template)
        replaceable = []
        for ini, end in inds_to_be_replaced:
            v = self.template[ini:end].split("_")
            if v[0] == 'template':
                template_name = v[1]
                args = {}
                if len(v) >= 2:
                    print(v)
                    for i in range(2,len(v)):
                        args[v[i].split(":")[0]] = v[i].split(":")[1]

                temp = PrototxtCosetes(map_template2file[template_name], {})
                val = temp.getString(args)
                replaceable.append(['<<'+self.template[ini:end]+'>>', val])
                
        #We replace here
        for to_b_repl, replace_with in replaceable:
            self.template = self.template.replace(to_b_repl, replace_with)

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
        for string_to_replace in fields_to_replace:
            aux = aux.replace('<<var_'+string_to_replace+'>>', fields_to_replace[string_to_replace])
        return aux

    def saveOutputPrototxt(self, out_filename, fields_to_replace):
        out_string = self.getString(fields_to_replace)
        f = open(out_filename, 'w')
        f.write(out_string)
        f.close()

if __name__ == '__main__':
    OUTPUT_FILENAME = '../ready.prototxt'
    OUTPUT_NEURONS = 1000

    variables_to_replace = {
        'LRMULTBASENET' : '0',
        'DEMULTBASENET' : '0',
        'LRMULTLASTLAYER' : '1',
        'DEMULTLASTLAYER' : '1',
        'OUTPUTNEURONS' : str(OUTPUT_NEURONS),
        'TRAINFILENAME': 'TRAIN-FILENAME',
        'VALFILENAME': 'VAL-FILENAME'
    }

    map_template2file = {
        'inputprototxt' :                   './base_network/my_network/base_files/input_layers_base/train_layers_base.prototxt',
        'evaltrainstage' :                  './base_network/my_network/base_files/output_layers_templates/final_output_base.prototxt',
        'crossentropylossintermediate' :    './base_network/my_network/base_files/output_layers_templates/crossentropylossintermediate.prototxt'
    }

    new_prototxt = PrototxtCosetes("./base_network/my_network/base_files/googlenetbase.prototxt", map_template2file)
    new_prototxt.saveOutputPrototxt(OUTPUT_FILENAME, variables_to_replace)


