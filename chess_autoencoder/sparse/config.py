from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
  config = config_dict.ConfigDict()

  config.model = config_dict.ConfigDict()
  config.train = config_dict.ConfigDict()

  config.model.latent_dim = 8
  config.model.embed_width = 8
  config.model.ln = False

  config.train.batch_size = 8
  #config.train.batch_size = config.get_ref('batch_size')
  config.train.shuffle = 0
  # config.train.optimizer = 'adamw'
  config.train.lr = 6e-4

  config.label = 'move' # or 'board'

  return config
