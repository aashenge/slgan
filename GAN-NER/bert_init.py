#### bert_init.py ####
import collections
import re
import tensorflow as tf


def limit_layer(name, start, end):
    if 'bert/encoder/layer' in name:
        layer_id_str = re.findall(r"bert/encoder/layer_(.+?)/", name)
        layer_id = int(layer_id_str)
        if start <= layer_id <= end:
            return True
        else:
            return False
    return True


def get_assignment_map_from_checkpoint(tvars, init_checkpoint, scope_head, start=0, end=999):
    """Compute the union of the current variables and checkpoint variables."""
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if len(scope_head) > 0:
            new_name = scope_head + '/' + name
        else:
            new_name = name
        if new_name not in name_to_variable:
            continue
        assignment_map[name] = new_name
        initialized_variable_names[new_name] = 1
        initialized_variable_names[new_name + ":0"] = 1

    return assignment_map, initialized_variable_names
