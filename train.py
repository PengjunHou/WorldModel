import pathlib
import subprocess
import sys
import warnings
from datetime import datetime

import ruamel.yaml as yaml

warnings.filterwarnings("ignore", ".*box bound precision lowered.*")
warnings.filterwarnings("ignore", ".*using stateful random seeds*")
warnings.filterwarnings("ignore", ".*is a deprecated alias for.*")

# directory = pathlib.Path(__file__)
# directory = directory.resolve()
# directory = directory.parent
# sys.path.append(str(directory.parent))
# sys.path.append(str(directory.parent.parent.parent))
# __package__ = directory.name
import simulator
from utils import Config, Flags, load_task_configs
# from simulator.run.train import Trainer

def main(argv=None):
    with open(r"config/model_configs/stgat.yaml", "r") as f:
        model_configs = yaml.YAML(typ="safe").load(f)
    # command line arguments
    parsed, other = Flags(task=["carla_comm"], actor_id=0, actors=0).parse_known(argv)
    config = Config({"stgat": model_configs["defaults"]})

    for name in parsed.task:
        print("Using task: ", name)
        env_config = load_task_configs(name, argv)
        config = config.update(env_config)
        env, config = simulator.create_task(config)
        
    confdir =  pathlib.Path("config")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config_filename = f"config_{timestamp}.yaml"
    config.save(str(confdir / config_filename))
    print(f"[Train] Config saved to {confdir / config_filename}")
    
    env.reset()
    time_steps = 0
    # agent = RandomAgent(env.obs_space, env.act_space, time_steps, config.V2X_test)
    # replay_buffer = ReplayBuffer(config.model.replay_buffer_size)
    
    while time_steps < 100000:
        action = env.action_space.sample()
        # obs = env.obs
        # action = agent.policy(env.obs, state=None, mode="train")
        next_obs, reward, done, _, info = env.step(action)
        time_steps += 1
        state = env.unwrapped.wrapper_obs()
        # print(f"[Train] Step: {time_steps}, Reward: {reward}, Done: {done}, state: {state}")
        if done:
            env.reset()
    print("[Train] Environment test run completed.")
    
    
    test_config = config.V2X_test
    step = 0
    # agent = BaseAgent(env.obs_space, env.act_space, step, test_config)
    # train(agent, env, test_config)
    
    # trainer = Trainer(env, config)
    # trainer.train()



if __name__ == "__main__":
    main()
