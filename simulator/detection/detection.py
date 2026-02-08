from typing import Callable, Dict, Tuple

import carla
from gymnasium import spaces

from ..carla_manager import WorldManager
from .detector import BaseDetector, CameraDetector, LidarDetector
from .utils import DETECTOR_DICT, DetectorType

SIMPLE_DETECTOR_NAME = "simple"


class Detection:
    """
    An observer is a collection of data detectors, each providing some observation data.

    The output of the Observer can be configured through ``env.observation.enabled``, which is passed to the constructor in ``obs_config``.

    In addtion, :py:meth:`register_simple_detector` provides a flexible way to supplement the observation data with a callback.
    """

    def __init__(self, world: WorldManager, obs_config: dict):
        self._world = world
        self._obs_config = obs_config
        self._data_detectors = self._init_data_detectors()

    def register_simple_detector(    # TODO
        self,
        key: str,
        observation_fn: Callable[[], Dict],
        observation_space: spaces.Space,
    ) -> None:
        """
        Register a simple observation function that may optionally use environment state.

        :param key: str, the key in observation space
        :param observation_fn: Callable[[], Dict], the callback function which returns the observation data
        :param observation_space: spaces.Space, the observation space
        """
        if SIMPLE_DETECTOR_NAME not in self._data_detectors:
            self._data_detectors[SIMPLE_DETECTOR_NAME] = SimpleDetector(self._world, {})
        self._data_detectors[SIMPLE_DETECTOR_NAME].register_observation(key, observation_fn, observation_space)

    def destroy(self) -> None:
        """Destroy all the registered detectors."""
        for detector in self._data_detectors.values():
            detector.destroy()

    def reset(self, ego: carla.Actor) -> None:
        """Reset all the registered detectors with the given ego vehicle."""
        for detector in self._data_detectors.values():
            print(f"[Observer] Resetting detector of type '{type(detector).__name__}'")
            detector.reset(ego)

    def _init_data_detectors(self) -> Dict[str, BaseDetector]:
        """Initialize the EndpointDetectors based on the observation configuration."""
        detectors = {}
        for name in self._obs_config.enabled:
            config = self._obs_config[name]
            detector_class = DETECTOR_DICT.get(DetectorType(config.detector))
            detector = detector_class(self._world, config)
            detectors[name] = detector
        return detectors

    def get_observation_space(self) -> spaces.Space:
        """Get the combined observation space from all the registered detectors."""
        obs_spaces = {}
        for detector in self._data_detectors.values():
            obs_spaces.update(detector.get_observation_space())
        return spaces.Dict(obs_spaces)

    def get_observation(self, env_state: Dict) -> Tuple[Dict, Dict]:
        """Get the current observation data from all the registered detectors."""
        obs = {}
        info = {}
        for detector in self._data_detectors.values():
            obs_data, info_data = detector.get_observation(env_state)
            obs.update(obs_data)
            info.update(info_data)
        return obs, info
