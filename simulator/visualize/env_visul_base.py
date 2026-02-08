
class EnvVisualBase:
    def __init__(self, config):
        self._config = config

    def render(self, mode="human"):
        """
        Override this method to implement custom rendering logic.
        The mode parameter can be "human" for on-screen display or "rgb_array" for returning an image array.
        """
        pass
    
    