"""
Launcher for all experiments. Download pre-training data, normalization statistics, and pre-trained checkpoints if needed.

"""

import os
import sys
import logging

import math
import hydra
from omegaconf import OmegaConf
# import gdown
# from download_url import (
#     get_dataset_download_url,
#     get_normalization_download_url,
#     get_checkpoint_download_url,
# )

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)



# add logger
LOG = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=os.path.join(
        os.getcwd(), "cfg"
    ),  # possibly overwritten by --config-path
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers will use the same time.
    OmegaConf.resolve(cfg)

    # run agent
    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    agent.run()


if __name__ == "__main__":
    #print(os.getcwd())
    main()
