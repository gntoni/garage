from collections import Iterable
from collections import namedtuple

import numpy as np
import tensorflow as tf


def compile_function(inputs, outputs, log_name=None):
    def run(*input_vals):
        sess = tf.get_default_session()
        return sess.run(outputs, feed_dict=dict(list(zip(inputs, input_vals))))

    return run


def flatten_batch(t, name=None):
    with tf.name_scope(name, "flatten_batch", [t]):
        shape = [-1] + list(t.shape[2:])
        return tf.reshape(t, shape)


def flatten_batch_dict(d, name=None):
    with tf.name_scope(name, "flatten_batch_dict", [d]):
        d_flat = dict()
        for k, v in d.items():
            d_flat[k] = flatten_batch(v)
        return d_flat


def filter_valids(t, valid, name=None):
    with tf.name_scope(name, "filter_valids", [t, valid]):
        return tf.boolean_mask(t, valid)


def filter_valids_dict(d, valid, name=None):
    with tf.name_scope(name, "filter_valids_dict", [d, valid]):
        d_valid = dict()
        for k, v in d.items():
            d_valid[k] = tf.boolean_mask(v, valid)
        return d_valid


def namedtuple_singleton(name, **kwargs):
    Singleton = namedtuple(name, kwargs.keys())
    return Singleton(**kwargs)


def flatten_inputs(deep):
    def flatten(deep):
        for d in deep:
            if isinstance(d, Iterable) and not isinstance(
                    d, (str, bytes, tf.Tensor, np.ndarray)):
                yield from flatten(d)
            else:
                yield d

    return list(flatten(deep))


def flatten_tensor_variables(ts):
    return tf.concat(
        axis=0,
        values=[tf.reshape(x, [-1]) for x in ts],
        name="flatten_tensor_variables")


def unflatten_tensor_variables(flatarr, shapes, symb_arrs):
    arrs = []
    n = 0
    for (shape, symb_arr) in zip(shapes, symb_arrs):
        size = np.prod(list(shape))
        arr = tf.reshape(flatarr[n:n + size], shape)
        arrs.append(arr)
        n += size
    return arrs


def new_tensor(name, ndim, dtype):
    return tf.placeholder(dtype=dtype, shape=[None] * ndim, name=name)


def new_tensor_like(name, arr_like):
    return new_tensor(name,
                      arr_like.get_shape().ndims, arr_like.dtype.base_dtype)


def concat_tensor_list(tensor_list):
    return np.concatenate(tensor_list, axis=0)


def concat_tensor_dict_list(tensor_dict_list):
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = concat_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = concat_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def stack_tensor_list(tensor_list):
    return np.array(tensor_list)
    # tensor_shape = np.array(tensor_list[0]).shape
    # if tensor_shape is tuple():
    #     return np.array(tensor_list)
    # return np.vstack(tensor_list)


def stack_tensor_dict_list(tensor_dict_list):
    """
    Stack a list of dictionaries of {tensors or dictionary of tensors}.
    :param tensor_dict_list: a list of dictionaries of {tensors or dictionary
     of tensors}.
    :return: a dictionary of {stacked tensors or dictionary of stacked tensors}
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = stack_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def split_tensor_dict_list(tensor_dict):
    keys = list(tensor_dict.keys())
    ret = None
    for k in keys:
        vals = tensor_dict[k]
        if isinstance(vals, dict):
            vals = split_tensor_dict_list(vals)
        if ret is None:
            ret = [{k: v} for v in vals]
        else:
            for v, cur_dict in zip(vals, ret):
                cur_dict[k] = v
    return ret


def to_onehot_sym(inds, dim):
    return tf.one_hot(inds, depth=dim, on_value=1, off_value=0)


def pad_tensor(x, max_len):
    return np.concatenate([
        x,
        np.tile(
            np.zeros_like(x[0]), (max_len - len(x), ) + (1, ) * np.ndim(x[0]))
    ])


def pad_tensor_n(xs, max_len):
    ret = np.zeros((len(xs), max_len) + xs[0].shape[1:], dtype=xs[0].dtype)
    for idx, x in enumerate(xs):
        ret[idx][:len(x)] = x
    return ret


def pad_tensor_dict(tensor_dict, max_len):
    keys = list(tensor_dict.keys())
    ret = dict()
    for k in keys:
        if isinstance(tensor_dict[k], dict):
            ret[k] = pad_tensor_dict(tensor_dict[k], max_len)
        else:
            ret[k] = pad_tensor(tensor_dict[k], max_len)
    return ret
