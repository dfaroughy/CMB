
import yaml
import ssl
import nltk
import numpy as np
from nltk.corpus import wordnet
from datetime import datetime

class Configs:
    def __init__(self, config_source):
        if isinstance(config_source, str):
            # If config_source is a file path, load the YAML file
            with open(config_source, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif isinstance(config_source, dict):
            # If config_source is a dict, use it directly
            config_dict = config_source
        else:
            raise ValueError("config_source must be a file path or a dictionary")
        
        # Recursively set attributes
        self._set_attributes(config_dict)
        
        if hasattr(self, 'data'):
            if hasattr(self.data, 'source') and hasattr(self.data, 'target'):
                self.general.experiment_name = f"{self.data.source.name}_to_{self.data.target.name}_{self.dynamics.name}_{self.model.name}"
                time = datetime.now().strftime("%Y.%m.%d_%Hh%M")
                self.general.experiment_name += f"_{time}"

    def _set_attributes(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Create a sub-config object
                sub_config = Configs(value)
                setattr(self, key, sub_config)
            else:
                setattr(self, key, value)

    def to_dict(self):
        """
        Recursively converts the Configs object into a dictionary.
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Configs):
                config_dict[key] = value.to_dict()
            else:
                config_dict[key] = value
        return config_dict

    def print(self):
        """
        Prints the configuration parameters in a structured format.
        """
        config_dict = self.to_dict()
        self._print_dict(config_dict)

    def _print_dict(self, config_dict, indent=0):
        """
        Helper method to recursively print the config dictionary.
        """
        for key, value in config_dict.items():
            prefix = ' ' * indent
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                self._print_dict(value, indent + 4)
            else:
                print(f"{prefix}{key}: {value}")

    def log_config(self, logger):
        """
        Logs the configuration parameters using the provided logger.
        """
        config_dict = self.to_dict()
        self._log_dict(config_dict, logger)

    def _log_dict(self, config_dict, logger, indent=0):
        """
        Helper method to recursively log the config dictionary.
        """
        for key, value in config_dict.items():
            prefix = ' ' * indent
            if isinstance(value, dict):
                logger.logfile.info(f"{prefix}{key}:")
                self._log_dict(value, logger, indent + 4)
            else:
                logger.logfile.info(f"{prefix}{key}: {value}")

def get_bigram(seed):
    """Return a random bigram of the form <adjective>_<noun>."""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # nltk.download("wordnet")  # Download WordNet data
    adjectives = [synset.lemmas()[0].name() for synset in wordnet.all_synsets(wordnet.ADJ)]
    nouns = [synset.lemmas()[0].name() for synset in wordnet.all_synsets(wordnet.NOUN)]
    adjectives = [adj for adj in adjectives if "-" not in adj and "_" not in adj and adj[0].islower() ]
    nouns = [noun for noun in nouns if "-" not in noun and "_" not in noun and noun[0].islower()]
    rng = np.random.default_rng(seed)
    i_adj, i_noun = rng.choice(len(adjectives)), rng.choice(len(nouns))

    # Return the bigram with the words capitalized

    return adjectives[i_adj].capitalize() + nouns[i_noun].capitalize()