import time
from Foofah.foofah_libs import operators as Operations


def create_python_prog(path, input_data=None, output_file=None):
    out_prog = ''
    # out_prog = "#\n# Synthetic Data Transformation Program\n#\n"
    # out_prog += "# Author:\n"
    # now = time.strftime("%c")
    # out_prog += "# Datetime: %s\n#\n\n" % (now)
    #
    # out_prog += "from foofah_libs.operators import *\n\n"

    # if input_data:
        # out_prog += "#\n# Raw Data\n#\n"
        # out_prog += "t = " + str(input_data) + "\n\n"

    # out_prog += "#\n# Data Transformation\n#\n"
    # print(input_data)

    Operations.PRUNE_1 = False

    for i, n in enumerate(reversed(path)):
        if i > 0:
            params = n.operation[2]
            params_to_apply = []
            out_prog += "t = " + n.operation[0]['name'] + "(t"
            # out_prog += n.operation[0]['name'] + ' '
            for i in range(1, n.operation[0]['num_params']):
                out_prog += ',' + params[i]
                param_to_add = params[i]
                if param_to_add.isnumeric():
                    param_to_add = int(param_to_add)
                params_to_apply += [param_to_add, ]
            # print('--')
            # print(n.operation[0]['name'] + str(params_to_apply))
            # print(n.operation[0]['fxn'](input_data, *params_to_apply))
            out_prog += ")\n"
            # out_prog = out_prog[:-1] + "\n"
    if output_file:
        fo = open("foo.txt", "w")
        fo.write(out_prog)
    # exit()
    return out_prog
