from ml_collections import config_dict

def get_config() -> config_dict.ConfigDict:
  config = config_dict.ConfigDict()

  config.model = config_dict.ConfigDict()
  config.train = config_dict.ConfigDict()

  config.model.latent_dim = 8
  config.model.embed_width = 8

  config.train.batch_size = 8
  config.train.shuffle = 0

  return config
