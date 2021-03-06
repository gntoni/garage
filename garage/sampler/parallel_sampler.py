import pickle

import numpy as np

from garage.misc import ext
from garage.misc import logger
from garage.misc import tensor_utils
from garage.sampler import SharedGlobal
from garage.sampler import singleton_pool
from garage.sampler.utils import rollout


def _worker_init(g, id):
    if singleton_pool.n_parallel > 1:
        import os
        os.environ['THEANO_FLAGS'] = 'device=cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    g.worker_id = id


def initialize(n_parallel):
    singleton_pool.initialize(n_parallel)
    singleton_pool.run_each(
        _worker_init, [(id, ) for id in range(singleton_pool.n_parallel)])


def _get_scoped_g(g, scope):
    if scope is None:
        return g
    if not hasattr(g, "scopes"):
        g.scopes = dict()
    if scope not in g.scopes:
        g.scopes[scope] = SharedGlobal()
        g.scopes[scope].worker_id = g.worker_id
    return g.scopes[scope]


def _worker_populate_task(g, env, policy, scope=None):
    g = _get_scoped_g(g, scope)
    g.env = pickle.loads(env)
    g.policy = pickle.loads(policy)


def _worker_terminate_task(g, scope=None):
    g = _get_scoped_g(g, scope)
    if getattr(g, "env", None):
        g.env.close()
        g.env = None
    if getattr(g, "policy", None):
        g.policy.terminate()
        g.policy = None


def populate_task(env, policy, scope=None):
    logger.log("Populating workers...")
    if singleton_pool.n_parallel > 1:
        singleton_pool.run_each(_worker_populate_task, [
            (pickle.dumps(env), pickle.dumps(policy), scope)
        ] * singleton_pool.n_parallel)
    else:
        # avoid unnecessary copying
        g = _get_scoped_g(singleton_pool.G, scope)
        g.env = env
        g.policy = policy
    logger.log("Populated")


def terminate_task(scope=None):
    singleton_pool.run_each(_worker_terminate_task,
                            [(scope, )] * singleton_pool.n_parallel)


def terminate():
    singleton_pool.terminate()


def _worker_set_seed(_, seed):
    logger.log("Setting seed to %d" % seed)
    ext.set_seed(seed)


def set_seed(seed):
    singleton_pool.run_each(_worker_set_seed,
                            [(seed + i, )
                             for i in range(singleton_pool.n_parallel)])


def _worker_set_policy_params(g, params, scope=None):
    g = _get_scoped_g(g, scope)
    g.policy.set_param_values(params)


def _worker_set_env_params(g, params, scope=None):
    g = _get_scoped_g(g, scope)
    g.env.set_param_values(params)


def _worker_collect_one_path(g, max_path_length, scope=None):
    g = _get_scoped_g(g, scope)
    path = rollout(g.env, g.policy, max_path_length)
    return path, len(path["rewards"])


def sample_paths(policy_params,
                 max_samples,
                 max_path_length=np.inf,
                 env_params=None,
                 scope=None):
    """
    :param policy_params: parameters for the policy. This will be updated on
     each worker process
    :param max_samples: desired maximum number of samples to be collected. The
     actual number of collected samples might be greater since all trajectories
     will be rolled out either until termination or until max_path_length is
     reached
    :param max_path_length: horizon / maximum length of a single trajectory
    :return: a list of collected paths
    """
    singleton_pool.run_each(
        _worker_set_policy_params,
        [(policy_params, scope)] * singleton_pool.n_parallel)
    if env_params is not None:
        singleton_pool.run_each(
            _worker_set_env_params,
            [(env_params, scope)] * singleton_pool.n_parallel)
    return singleton_pool.run_collect(
        _worker_collect_one_path,
        threshold=max_samples,
        args=(max_path_length, scope),
        show_prog_bar=True)


def truncate_paths(paths, max_samples):
    """
    Truncate the list of paths so that the total number of samples is exactly
    equal to max_samples. This is done by removing extra paths at the end of
    the list, and make the last path shorter if necessary
    :param paths: a list of paths
    :param max_samples: the absolute maximum number of samples
    :return: a list of paths, truncated so that the number of samples adds up
    to max-samples
    """
    # chop samples collected by extra paths
    # make a copy
    paths = list(paths)
    total_n_samples = sum(len(path["rewards"]) for path in paths)
    while paths and total_n_samples - len(paths[-1]["rewards"]) >= max_samples:
        total_n_samples -= len(paths.pop(-1)["rewards"])
    if paths:
        last_path = paths.pop(-1)
        truncated_last_path = dict()
        truncated_len = len(
            last_path["rewards"]) - (total_n_samples - max_samples)
        for k, v in last_path.items():
            if k in ["observations", "actions", "rewards"]:
                truncated_last_path[k] = tensor_utils.truncate_tensor_list(
                    v, truncated_len)
            elif k in ["env_infos", "agent_infos"]:
                truncated_last_path[k] = tensor_utils.truncate_tensor_dict(
                    v, truncated_len)
            else:
                raise NotImplementedError
        paths.append(truncated_last_path)
    return paths
