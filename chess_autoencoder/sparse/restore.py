import jax
import orbax.checkpoint as ocp
from jax.tree_util import tree_flatten, tree_structure, tree_unflatten

#c_options = ocp.CheckpointManagerOptions(max_to_keep=3)
c_path = '/tmp/logdir/checkpoint'
c_mngr = ocp.CheckpointManager(c_path)
print('step: ', c_mngr.latest_step())
restored = c_mngr.restore(
  c_mngr.latest_step(),
  args=ocp.args.StandardRestore())
#print(jax.tree_map(lambda x: x.shape, restored) )
print()
print(tree_structure(restored))

print()

def show_example(structured):
  flat, tree = tree_flatten(structured)
  unflattened = tree_unflatten(tree, flat)
  print(f"{structured=}\n  {flat=}\n  {tree=}\n  {unflattened=}")

for foo in restored:
  show_example(foo)
