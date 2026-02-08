from .config import Config
from .flags import Flags

def load_task_configs(task_name: str, argv = None):
    """
    Load the task configs for the specified task name.
    The task name should be one of the keys in the ``tasks.yaml`` file.

    :param task_name: str, the name of the task

    :return: the task configs
    """
    import os

    import yaml

    dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
    with open(os.path.join(dir, "common.yaml")) as f:
        config = yaml.safe_load(f)
        config = Config(config)
    with open(os.path.join(dir, "tasks.yaml")) as f:
        task_config = yaml.safe_load(f)
        config = config.update(task_config[task_name])
        
    if argv is not None:
         config, _ = Flags(config).parse_known(argv)
    return config