import time
from typing import Any, Callable, Dict

import flax
import jax
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
import tensorflow as tf
from chex import assert_rank, assert_shape
from flax.training import common_utils, train_state

from model import AutoEncoder
from schema import (LABEL_VOCABULARY, TRANSFORMER_FEATURES, TRANSFORMER_LENGTH,
                    TRANSFORMER_SHAPE, TRANSFORMER_VOCABULARY)


def old_compute_metrics(*, logits: jnp.ndarray, labels: jnp.ndarray) -> Dict[str, jnp.ndarray]:
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'accuracy': accuracy,
  }
  return metrics


def old_accumulate_metrics(metrics: list) -> dict:
  metrics = jax.device_get(metrics)
  return {
    k: np.mean([metric[k] for metric in metrics])
    for k in metrics[0]
  }

@jax.jit
def old_train_step(
    state: train_state.TrainState,
    x: jnp.ndarray,
    label: jnp.ndarray
):
  def _loss_fn(params):
    logits = state.apply_fn({'params': params}, x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=label)
    return loss.mean(), logits

  gradient_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  (loss, logits), grads = gradient_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = old_compute_metrics(logits=logits, labels=label)
  metrics['loss'] = loss
  return state, metrics


class TestModel:

  def test_loss_decreases(self):
    latent_dim = 2
    embed_width = 3
    rng = jax.random.PRNGKey(42)
    rng, rnd_ae = jax.random.split(rng, num=2)
    ae = AutoEncoder(latent_dim=latent_dim, embed_width=embed_width)
    sample_x = jax.random.randint(key=rng,
                                  shape=((1,) + TRANSFORMER_SHAPE),
                                  minval=0,
                                  maxval=TRANSFORMER_VOCABULARY,
                                  dtype=jnp.int32)
    ae_variables = ae.init(rnd_ae, sample_x)

    labels = jnp.array([[724]])
    print('labels: ', labels)
    optimizer = optax.adamw(learning_rate=0.1)
    ts_ae = train_state.TrainState.create(
      apply_fn=ae.apply,
      tx=optimizer,
      params=ae_variables['params'])

    first = None
    for _ in range(20):
      ts_ae, metrics = old_train_step(ts_ae, sample_x, labels)
      m = old_accumulate_metrics([metrics])
      if first is None:
        first = m
      print('acc: ', m)
    last = m
    assert last['accuracy'] > 0.9
    assert last['loss'] < first['loss']
    assert last['loss'] < 0.1


  def test_logits_shape(self):
    latent_dim = 2
    embed_width = 3
    rng = jax.random.PRNGKey(42)
    rng, rnd_ae = jax.random.split(rng, num=2)
    ae = AutoEncoder(latent_dim=latent_dim, embed_width=embed_width)
    sample_x = jax.random.randint(key=rng,
                                  shape=((1,) + TRANSFORMER_SHAPE),
                                  minval=0,
                                  maxval=TRANSFORMER_VOCABULARY,
                                  dtype=jnp.int32)
    ae_variables = ae.init(rnd_ae, sample_x)

    logits = ae.apply(ae_variables, sample_x)
    assert_shape(logits, (None, TRANSFORMER_LENGTH, LABEL_VOCABULARY))
