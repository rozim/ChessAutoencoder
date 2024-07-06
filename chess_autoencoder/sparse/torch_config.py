
from ml_collections.config_dict import ConfigDict

from absl import app

def get_config() -> ConfigDict:
  c = ConfigDict()
  c.model = ConfigDict()
  c.train = ConfigDict()
  c.test = ConfigDict()
  c.opt = ConfigDict()

  c.opt_type = 'SGD'
  c.opt.lr = 1e-1


  c.model_type = 'Simple'
  c.model.embed_dim = 16
  c.model.layer_width = 64

  c.batch_size = 64

  c.train_epochs = 100
  c.train_steps = 10 # within epochs, before progress output

  c.test_epochs = 10

  return c


def main(argv):
  print(get_config())

if __name__ == '__main__':
  app.run(main)
