# config/settings.py
# Loads configuration from config.yaml and provides them as a singleton object.

import yaml

class Settings:
    """
    A singleton class to manage application settings.
    It loads configuration from a YAML file and provides easy access to the settings.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Settings, cls).__new__(cls)
        return cls._instance

    def __init__(self, domain=None, task=None, config_path='config/config.yaml'):
        # The __init__ will be called every time Settings() is invoked,
        # but the instance will be the same. We only load the config once.
        if not hasattr(self, 'loaded'):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.loaded = True

        if domain:
            self.domain = domain
        if task:
            self.task = task
            
    def get(self, key, default=None):
        """
        Retrieves a value from the configuration.
        """
        return self.config.get(key, default)

    @property
    def api_keys(self):
        return self.get('api_keys', {})

    @property
    def paths(self):
        return self.get('paths', {})

    @property
    def timeouts(self):
        return self.get('timeouts', {})

    @property
    def domain_config(self):
        """
        Returns the configuration for the currently set domain.
        """
        domains = self.get('domains', {})
        return domains.get(self.domain, domains.get('default', {}))

# You can create a default instance for easy import elsewhere
settings = Settings()
