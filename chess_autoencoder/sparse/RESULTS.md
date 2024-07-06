
latent=8, embed=8, batch=8: peak acc			0.03

time python train.py  --config.model.latent_dim=16	acc 0.032

time python train.py  --config.model.latent_dim=16 --config.model.embed_width=16 acc 0.040
