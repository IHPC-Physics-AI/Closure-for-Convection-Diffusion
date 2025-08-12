import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
import time

cur_dir = os.getcwd()
print(cur_dir)
#dir = '/home/ooicc/ARIA'
dir = cur_dir
sys.path.append(dir)

import numpy as np
import scipy
import random
import matplotlib.pyplot as plt

import jax
import optax
from flax import linen as nn
from jax import numpy as jnp
from flax.training import train_state  # Useful dataclass to keep train state
import flax

from functools import partial
import pickle

from numerical_methods import physics, ssim

## Dataset
dx, dy = 0.5, 0.5
ny, nx = 26, 49
dt = 0.4
NSUBSTEPS = 4

with open(f'{dir}/dataset/dataset_v2.pickle', 'rb') as handle:
  dataset = pickle.load(handle)

train_set = [0, 2, 4, 5, 6, 7, 8, 10, 12, 13, 15, 17, 19, 21, 23, 11, 14, 16, 25, 37, 38, 41, 42, 43, 49, 50]

INFLOW_LOCS = dataset['INFLOW_LOCS'][train_set,]
FLOW_TIMES = [jnp.concatenate((jnp.zeros((1,)),jnp.array(dataset['FLOW_TIMES'][i]))) for i in train_set]
SMOKE_FIELD = [jnp.concatenate((jnp.zeros((1,ny,nx)), jnp.array(dataset['SMOKE_FIELD'][i])), axis=0) for i in train_set]
VELOCITY = dataset['VELOCITY']
REL_LOC = dataset['rel_loc']
TERRAIN = dataset['TERRAIN']
SPLINE_TCK = [dataset['SPLINE_TCK'][i] for i in train_set]

INFLOW_VALS = []
for set_idx in range(len(SMOKE_FIELD)):
  y,x = INFLOW_LOCS[set_idx]
  inflow_for_set = []
  for i in range(len(REL_LOC)):
    rel_y, rel_x = REL_LOC[i]
    inflow_for_set_for_loc = []
    for nt in range(1, 6001):
      val = scipy.interpolate.splev(nt*dt, SPLINE_TCK[set_idx][i])
      inflow_for_set_for_loc.append(val)
    inflow_for_set.append(inflow_for_set_for_loc)
  INFLOW_VALS.append(inflow_for_set)

INFLOW_VALS = jnp.array(INFLOW_VALS)

FILENAME_PREFIX = f'{dir}/expt3_markovian'

diff_x_mask = jnp.logical_and(TERRAIN[:,1:],TERRAIN[:,:-1])
diff_y_mask = jnp.logical_and(TERRAIN[1:,:],TERRAIN[:-1,:])

diff_x_mask = diff_x_mask.astype(jnp.float32)
diff_y_mask = diff_y_mask.astype(jnp.float32)

MAX_STEPS = int((60+1e-3)/dt)

## Turbulent Diffusivity
import dataset.read_data_to_grid as rdtg

grid = rdtg.read_data(f"{dir}/dataset/turb_vis.txt")
turb_vis = rdtg.extract_xyz_to_array(grid, x_range=(0, 48), y_range=(0, 25), z_range=(1,1), yxz=True)
turb_diff = jnp.array(turb_vis/0.7)

turb_diff_x = ((turb_diff[:,1:] + turb_diff[:,:-1])/2).squeeze()
turb_diff_y = ((turb_diff[1:,:] + turb_diff[:-1,:])/2).squeeze()

turb_diff_x = jnp.where(diff_x_mask, turb_diff_x, 0)
turb_diff_y = jnp.where(diff_y_mask, turb_diff_y, 0)

## Neural Networks
class ClosureNet(nn.Module):
    def setup(self):
        # for transforming the output
        self.conv1 = nn.Conv(features=32, kernel_size=(3,3), strides=1, padding='SAME', kernel_init=nn.initializers.xavier_uniform())
        self.conv2 = nn.Conv(features=32, kernel_size=(3,3), strides=1, padding='SAME', kernel_init=nn.initializers.xavier_uniform())
        self.conv3 = nn.Conv(features=1, kernel_size=(3,3), strides=1, padding='SAME', kernel_init=nn.initializers.xavier_uniform())


    def __call__(self, x):
        output = x
        output = nn.leaky_relu(self.conv1(output), negative_slope=0.01) + jnp.pad(output, ((0,0),(0,0),(22,0)))
        output = nn.leaky_relu(self.conv2(output), negative_slope=0.01) + output
        output = self.conv3(output)
        return output

class InflowNet(nn.Module):
    def setup(self):
        self.conv1 = nn.Conv(features=8, kernel_size=(3,3), strides=1, padding='SAME', kernel_init=nn.initializers.xavier_uniform())
        self.conv2 = nn.Conv(features=16, kernel_size=(3,3), strides=1, padding='SAME', kernel_init=nn.initializers.xavier_uniform())
        self.conv3 = nn.Conv(features=32, kernel_size=(3,3), strides=1, padding='SAME', kernel_init=nn.initializers.xavier_uniform())
        self.conv4 = nn.Conv(features=64, kernel_size=(3,3), strides=1, padding='SAME', kernel_init=nn.initializers.xavier_uniform())

        self.fc1 = nn.Dense(features=64, kernel_init=nn.initializers.xavier_uniform())
        self.fc2 = nn.Dense(features=64, kernel_init=nn.initializers.xavier_uniform())
        self.fc3 = nn.Dense(features=4, kernel_init=nn.initializers.xavier_uniform())

    def __call__(self, x):
        # # Encoder
        enc1 = self.conv1(x)                                  # (26,49)
        enc1 = nn.leaky_relu(enc1, negative_slope=0.01)       # (26,49)

        enc2 = nn.max_pool(enc1, (2, 3), strides=(2, 3))      # (13,16)
        enc2 = self.conv2(enc2)                               # (13,16)
        enc2 = nn.leaky_relu(enc2, negative_slope=0.01)       # (13,16)

        enc3 = nn.max_pool(enc2, (2, 3), strides=(2, 3))      # (6,5)
        enc3 = self.conv3(enc3)                               # (6,5)
        enc3 = nn.leaky_relu(enc3, negative_slope=0.01)       # (6,5)

        enc4 = nn.max_pool(enc3, (2, 2), strides=(2, 2))      # (3,2)
        enc4 = self.conv4(enc4)                               # (3,2)
        enc4 = nn.leaky_relu(enc4, negative_slope=0.01)       # (3,2)

        bottleneck = nn.max_pool(enc4, (3, 2), strides=(3, 2))            # (1,1)

        output = nn.leaky_relu(self.fc1(bottleneck.squeeze()))
        output = nn.leaky_relu(self.fc2(output))
        output = self.fc3(output)
        return 10**(-5+jnp.tanh(output))

x = jnp.zeros((ny, nx, 10))
tabulate_fn = nn.tabulate(ClosureNet(), jax.random.PRNGKey(0), console_kwargs={"width": 120})

print(tabulate_fn(x))

## Phase 1
closure_net = ClosureNet()
inflow_net = InflowNet()

### Simulation Function
@jax.jit
def conv_diff_single_step(params,
                          smoke_initial: jnp.array,
                          hidden_state: jnp.array,
                          cell_state: jnp.array,
                          velocity: jnp.array,
                          time_curr: float,
                          inflow_loc: jnp.array,
                          inflow_vals: jnp.array,
                          terrain: jnp.array,
                          dt: float):
    y,x = inflow_loc
    inflow_marker = jnp.zeros((26,49))
    for i in range(len(REL_LOC)):
        rel_y, rel_x = REL_LOC[i]

        # set smoke at inflow locations
        inflow_marker = inflow_marker.at[y+rel_y, x+rel_x].set(1.0)

    # compute inflow term
    input = jnp.stack((terrain,
                       velocity[0,:,:],
                       velocity[1,:,:],
                       velocity[2,:,:],
                       jnp.pad(params['diffusivity_x'], ((0, 0), (1, 0))),
                       jnp.pad(params['diffusivity_y'], ((1, 0), (0, 0))),
                       inflow_marker), axis=-1) #(4,)
    inflow = inflow_net.apply({'params': params['inflow']}, input)
    inflow_term = jnp.zeros((26,49))
    for i in range(len(REL_LOC)):
        rel_y, rel_x = REL_LOC[i]
        inflow_term = inflow_term.at[y+rel_y, x+rel_x].set(inflow[i])

    smoke_pred = smoke_initial
    for _ in range(NSUBSTEPS):
      advection_step = physics.advect_fvm(field=smoke_pred,
                                              velocity=velocity,
                                              dx=dx,
                                              dy=dy) * dt/NSUBSTEPS

      diffusion_step = physics.diffuse_2d_fvm(field=smoke_pred,
                                              diffusivity_x=params['diffusivity_x'],
                                              diffusivity_y=params['diffusivity_y'],
                                              dx=dx,
                                              dy=dy) * dt/NSUBSTEPS

      inflow_step = inflow_term * dt/NSUBSTEPS
      smoke_pred = smoke_pred + advection_step + diffusion_step + inflow_step

    # compute closure term
    smoke_mean = jnp.mean(smoke_initial)
    eps = 1e-8
    smoke_std = jnp.std(smoke_initial)
    smoke_std_safe = jnp.where(smoke_std > eps, smoke_std, eps)
    input = jnp.stack(((smoke_initial-smoke_mean)/smoke_std_safe,
                        terrain * smoke_mean,
                        terrain * smoke_std,
                        velocity[0,:,:],
                        velocity[1,:,:],
                        velocity[2,:,:],
                        jnp.pad(params['diffusivity_x'], ((0, 0), (1, 0))),
                        jnp.pad(params['diffusivity_y'], ((1, 0), (0, 0))),
                        inflow_term,
                        terrain), axis=-1)  # (26,49,8)

    # (26,49,1)
    output = closure_net.apply({'params': params['closure']}, input)
    # print(jnp.mean(output), jnp.std(output))
    closure_term = output * 1e-10        # a denormalization value, from previous tests
    smoke_pred = (smoke_pred\
                    + closure_term.squeeze()) * terrain

    time_next = time_curr + dt

    # smoke_pred = jnp.maximum(smoke_pred, 0.0)

    # no updates to hidden states
    hidden_pred, cell_pred = hidden_state, cell_state

    return (smoke_pred, hidden_pred, cell_pred, time_next, params, inflow_loc, inflow_vals), smoke_pred

### Training Function
def scan_body(carry, _):
    (smoke, hidden, cell, time, params, inflow_loc, inflow_vals, step_idx, nsteps) = carry

    do_step = step_idx < nsteps
    (smoke_next, hidden_next, cell_next, time_next, _, _, _), _ = conv_diff_single_step(
        params, smoke, hidden, cell, VELOCITY, time, inflow_loc, inflow_vals, TERRAIN, dt
    )

    # Select updated state only if we're still under nsteps
    smoke = jax.lax.select(do_step, smoke_next, smoke)
    hidden = jax.lax.select(do_step, hidden_next, hidden)
    cell = jax.lax.select(do_step, cell_next, cell)
    time = jax.lax.select(do_step, time_next, time)

    # smoke_to_chain = jax.lax.select(do_step, smoke_next, jnp.zeros((ny,nx)))

    return (smoke, hidden, cell, time, params, inflow_loc, inflow_vals, step_idx + 1, nsteps), smoke

@jax.jit
def conv_diff_nsteps(params, smoke_initial, hidden_initial, cell_initial, time_initial,
                     inflow_loc, inflow_vals, nsteps):
    carry = (smoke_initial, hidden_initial, cell_initial, time_initial,
             params, inflow_loc, inflow_vals, 0, nsteps)
    (smoke_final, hidden_final, cell_final, _, _, _, _, _, _), smoke_pred_chain = \
        jax.lax.scan(scan_body, carry, xs=None, length=MAX_STEPS)
    return smoke_final, hidden_final, cell_final, smoke_pred_chain

conv_diff_nstep_vmap = jax.vmap(conv_diff_nsteps, in_axes=(None, 0, 0, 0, 0, 0, 0, 0), out_axes=(0, 0, 0, 0))
stagger_gradient_vmap = jax.vmap(physics.stagger_gradient, in_axes=(0, None, None), out_axes=(0, 0))

@jax.jit
def l2_loss(x, alpha):
  return alpha * (x ** 2).sum()

@jax.jit
def loss_fn(params,
            smoke_initial_all,
            hidden_initial_all,
            cell_initial_all,
            time_initial_all,
            nsteps_all,
            inflow_loc_all,
            inflow_vals_all,
            smoke_target_all,
            mask_target_all,
            thresh):
    smoke_pred_all, hidden_final_all, cell_final_all, smoke_pred_chain = conv_diff_nstep_vmap(params,
                                                                                              smoke_initial_all,
                                                                                              hidden_initial_all,
                                                                                              cell_initial_all,
                                                                                              time_initial_all,
                                                                                              inflow_loc_all,
                                                                                              inflow_vals_all,
                                                                                              nsteps_all)
    loss1 = (optax.l2_loss(smoke_pred_chain, smoke_target_all)/(smoke_target_all+1e-9)**2)*mask_target_all[:, :, None, None]
    loss1 = jnp.sum(jnp.mean(loss1, axis=(2,3)))

    loss2 = sum(
                l2_loss(w, alpha=1e-6)
                for w in jax.tree_util.tree_leaves(params['closure'])
            )
    loss3 = jnp.sum((smoke_pred_chain[:,1:] - smoke_pred_chain[:,:-1])**2)*1e4

    loss = loss1+loss2+loss3
    return loss, (smoke_pred_all, hidden_final_all, cell_final_all, loss1, loss2, loss3)

def train_step(state,
              batch_size:int,
              smoke_initial_set:list,
              hidden_initial_set:list,
              cell_initial_set:list,
              time_initial_set:list,
              dt:float,
              thresh:float=1e-9):

    grads_batch = None
    loss_batch = 0.0
    loss1_batch = 0.0

    batch_count = 0

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    jit_grad_fn = jax.jit(grad_fn)

    grads_all = None

    nsim = []
    while batch_count < batch_size:
      # list of indices that has not yet reached the final timestep
      valid_sets = [i for i in range(len(SMOKE_FIELD)) if time_initial_set[i]<len(FLOW_TIMES[i])-1]

      # indices of smoke states that should be propagated
      to_sample = min(len(valid_sets), batch_size-batch_count)
      nsim.append(to_sample)
      set_indices = random.sample(valid_sets, min(len(valid_sets), to_sample))
      batch_count += len(set_indices)

      # suppose s = len(set_indices)
      smoke_initial_all = jnp.array([smoke_initial_set[set_idx] for set_idx in set_indices])    # (s,ny,nx)
      hidden_initial_all = jnp.array([hidden_initial_set[set_idx] for set_idx in set_indices])  # (s,256)
      cell_initial_all = jnp.array([cell_initial_set[set_idx] for set_idx in set_indices])      # (s,256)
      time_initial_all = jnp.array([FLOW_TIMES[set_idx][time_initial_set[set_idx]] for set_idx in set_indices]) # (s,)
      inflow_loc_all = jnp.array([INFLOW_LOCS[set_idx] for set_idx in set_indices])
      inflow_vals_all = jnp.array([INFLOW_VALS[set_idx] for set_idx in set_indices])

      # simulate to the next 60s
      time_final_all = jnp.array([int((FLOW_TIMES[set_idx][time_initial_set[set_idx]]+60.0)/60.0)*60.0 for set_idx in set_indices])  # (s,)
      time_final_idx = [list(FLOW_TIMES[set_idx]).index(time_final_all[i]) for i,set_idx in enumerate(set_indices)]

      nsteps_all = ((time_final_all - time_initial_all + 1e-3)/dt).astype(int)  # (s,)

      smoke_target_all = jnp.zeros((len(set_indices), MAX_STEPS, ny, nx))   # (s, MAX_STEPS, ny, nx)
      mask_target_all = jnp.zeros((len(set_indices), MAX_STEPS))            # (s, MAX_STEPS)
      for i, set_idx in enumerate(set_indices):     # i from 0 to s-1
        for j in range(time_initial_set[set_idx]+1, time_final_idx[i]+1):   # look at all the indices which has a desired SMOKE_FIELD
          t_shifted_idx = int((FLOW_TIMES[set_idx][j]-FLOW_TIMES[set_idx][time_initial_set[set_idx]]+1e-3)/dt)
          smoke_target_all = smoke_target_all.at[i,t_shifted_idx].set(SMOKE_FIELD[set_idx][j])
          mask_target_all = mask_target_all.at[i,t_shifted_idx].set(1.0)

      (loss, (smoke_pred_all, hidden_pred_all, cell_final_all, loss1, loss2, loss3)), grads = grad_fn(state.params,
                                                                                smoke_initial_all,
                                                                                hidden_initial_all,
                                                                                cell_initial_all,
                                                                                time_initial_all,
                                                                                nsteps_all,
                                                                                inflow_loc_all,
                                                                                inflow_vals_all,
                                                                                smoke_target_all,
                                                                                mask_target_all,
                                                                                thresh)

      if grads_all == None:
        grads_all = grads
      else:
        grads_all = jax.tree_util.tree_map(lambda x, y: x + y, grads_all, grads)

      # update the set of initial values with the predicted values
      for i,set_idx in enumerate(set_indices):
        smoke_initial_set[set_idx] = smoke_pred_all[i]
        hidden_initial_set[set_idx] = hidden_pred_all[i]
        cell_initial_set[set_idx] = cell_final_all[i]
        time_initial_set[set_idx] = time_final_idx[i]

    grads_all = jax.tree_util.tree_map(lambda x: x/batch_size, grads_all)

    ## gradient clipping
    clipper = optax.clip_by_global_norm(0.1)
    clip_state = clipper.init(grads_all)
    new_grads, clip_state = clipper.update(grads_all, clip_state)

    state = state.apply_gradients(grads=new_grads)

    state.params['diffusivity_x'] = state.params['diffusivity_x'] * diff_x_mask
    state.params['diffusivity_y'] = state.params['diffusivity_y'] * diff_y_mask
    state.params['diffusivity_x'] = jnp.maximum(state.params['diffusivity_x'], 0.0)
    state.params['diffusivity_y'] = jnp.maximum(state.params['diffusivity_y'], 0.0)

    return state, loss, smoke_initial_set, hidden_initial_set, cell_initial_set, time_initial_set, loss1, loss2, loss3, nsim

def create_train_state(params_nn, params_physics, learning_rate):
    params = {}
    params['closure'] = params_nn['closure']
    params['inflow'] = params_nn['inflow']
    params['diffusivity_x'] = params_physics['diffusivity_x']
    params['diffusivity_y'] = params_physics['diffusivity_y']

    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=conv_diff_nsteps,
                                         params=params,
                                         tx=tx)
    
### Phase 1.1 Training
# 500 epochs
first_larger_index = [jnp.searchsorted(dataset['FLOW_TIMES'][i], 60.0, side='right') for i in range(len(dataset['FLOW_TIMES']))]
FLOW_TIMES = [jnp.array(dataset['FLOW_TIMES'][i][:first_larger_index[i]]) for i in train_set]
SMOKE_FIELD = [jnp.array(dataset['SMOKE_FIELD'][i][:first_larger_index[i]]) for i in train_set]

params_nn = {}
params_nn['closure'] = closure_net.init(jax.random.PRNGKey(1), jnp.zeros((ny, nx, 10)))['params']
params_nn['inflow'] = inflow_net.init(jax.random.PRNGKey(1), jnp.zeros((1, ny, nx, 7)))['params']
params_physics = {}
params_physics['diffusivity_x'] = turb_diff_x
params_physics['diffusivity_y'] = turb_diff_y

state = create_train_state(params_nn=params_nn, params_physics=params_physics,learning_rate=1e-5)

BATCH_SIZE = 5
ITR_PER_EPOCH = len(FLOW_TIMES)*1

losses = []
min_loss = np.inf
for epoch in range(0,500):
    epoch_loss = 0.0
    data_loss = 0.0

    start = time.time()
    hidden_curr_set = []
    cell_curr_set = []
    smoke_curr_set = []
    for set_idx in range(len(SMOKE_FIELD)):
        smoke_curr_set.append(jnp.zeros((ny,nx)))
        hidden_curr_set.append(jnp.zeros((4,128)))
        cell_curr_set.append(jnp.zeros((4,128)))

    time_curr_set = [0 for _ in range(len(SMOKE_FIELD))]

    itr = 0
    while itr < ITR_PER_EPOCH:
        itr_batch_size = min(BATCH_SIZE, ITR_PER_EPOCH-itr)
        state, loss, smoke_curr_set, hidden_curr_set, cell_curr_set, time_curr_set, loss1, loss2, loss3, nsim = \
            train_step(state,
                        batch_size=itr_batch_size,
                        smoke_initial_set=smoke_curr_set,
                        hidden_initial_set=hidden_curr_set,
                        cell_initial_set=cell_curr_set,
                        time_initial_set=time_curr_set,
                        dt=dt,
                        thresh=1e-8)

        itr += BATCH_SIZE
        print(f"EPOCH {epoch}, batch size: {itr_batch_size}, nsim={nsim} loss={loss:.4f}, rel loss={loss1:.4f}, reg loss={loss2}, smooth loss={loss3}")
        epoch_loss += loss
        data_loss += loss1
    # saving every 100 epochs
    if epoch%100==99:
        state_dict = flax.serialization.to_state_dict(state)
        with open(f'{FILENAME_PREFIX}_epoch_{epoch}.pickle', 'wb') as handle:
            pickle.dump(state_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if epoch_loss < min_loss:
        state_dict = flax.serialization.to_state_dict(state)
        with open(f'{FILENAME_PREFIX}_min.pickle', 'wb') as handle:
            pickle.dump(state_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        min_loss = epoch_loss
        print("New minimum achieved")

    losses.append(epoch_loss)
    end = time.time()
    print(f"EPOCH {epoch} (time={(end-start):.1f}): loss={epoch_loss:.4f}, data_loss={data_loss:.4f}")

jax.clear_caches()

### Phase 1.2 Training
# 500 epochs
first_larger_index = [jnp.searchsorted(dataset['FLOW_TIMES'][i], 120.0, side='right') for i in range(len(dataset['FLOW_TIMES']))]
FLOW_TIMES = [jnp.array(dataset['FLOW_TIMES'][i][:first_larger_index[i]]) for i in train_set]
SMOKE_FIELD = [jnp.array(dataset['SMOKE_FIELD'][i][:first_larger_index[i]]) for i in train_set]

with open(f'{FILENAME_PREFIX}_epoch_499.pickle', 'rb') as bunch:
  state_dict = pickle.load(bunch)
params_nn = {}
params_nn['closure'] = state_dict['params']['closure']
params_nn['inflow'] = state_dict['params']['inflow']
params_physics = {}
params_physics['diffusivity_x'] = state_dict['params']['diffusivity_x']
params_physics['diffusivity_y'] = state_dict['params']['diffusivity_y']
state = create_train_state(params_nn=params_nn, params_physics=params_physics,learning_rate=1e-5)

BATCH_SIZE = 5
ITR_PER_EPOCH = len(FLOW_TIMES)*2

losses = []
min_loss = np.inf
for epoch in range(500,1000):
    epoch_loss = 0.0
    data_loss = 0.0

    start = time.time()
    hidden_curr_set = []
    cell_curr_set = []
    smoke_curr_set = []
    for set_idx in range(len(SMOKE_FIELD)):
        smoke_curr_set.append(jnp.zeros((ny,nx)))
        hidden_curr_set.append(jnp.zeros((4,128)))
        cell_curr_set.append(jnp.zeros((4,128)))

    time_curr_set = [0 for _ in range(len(SMOKE_FIELD))]

    itr = 0
    while itr < ITR_PER_EPOCH:
        itr_batch_size = min(BATCH_SIZE, ITR_PER_EPOCH-itr)
        state, loss, smoke_curr_set, hidden_curr_set, cell_curr_set, time_curr_set, loss1, loss2, loss3, nsim = \
            train_step(state,
                        batch_size=itr_batch_size,
                        smoke_initial_set=smoke_curr_set,
                        hidden_initial_set=hidden_curr_set,
                        cell_initial_set=cell_curr_set,
                        time_initial_set=time_curr_set,
                        dt=dt,
                        thresh=1e-8)

        itr += BATCH_SIZE
        print(f"EPOCH {epoch}, batch size: {itr_batch_size}, nsim={nsim} loss={loss:.4f}, rel loss={loss1:.4f}, reg loss={loss2}, smooth loss={loss3}")
        epoch_loss += loss
        data_loss += loss1
    # saving every 100 epochs
    if epoch%100==99:
        state_dict = flax.serialization.to_state_dict(state)
        with open(f'{FILENAME_PREFIX}_epoch_{epoch}.pickle', 'wb') as handle:
            pickle.dump(state_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if epoch_loss < min_loss:
        state_dict = flax.serialization.to_state_dict(state)
        with open(f'{FILENAME_PREFIX}_min.pickle', 'wb') as handle:
            pickle.dump(state_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        min_loss = epoch_loss
        print("New minimum achieved")

    losses.append(epoch_loss)
    end = time.time()
    print(f"EPOCH {epoch} (time={(end-start):.1f}): loss={epoch_loss:.4f}, data_loss={data_loss:.4f}")

jax.clear_caches()

### Phase 1.3 Training
# 500 epochs
first_larger_index = [jnp.searchsorted(dataset['FLOW_TIMES'][i], 300.0, side='right') for i in range(len(dataset['FLOW_TIMES']))]
FLOW_TIMES = [jnp.array(dataset['FLOW_TIMES'][i][:first_larger_index[i]]) for i in train_set]
SMOKE_FIELD = [jnp.array(dataset['SMOKE_FIELD'][i][:first_larger_index[i]]) for i in train_set]

with open(f'{FILENAME_PREFIX}_epoch_999.pickle', 'rb') as bunch:
  state_dict = pickle.load(bunch)
params_nn = {}
params_nn['closure'] = state_dict['params']['closure']
params_nn['inflow'] = state_dict['params']['inflow']
params_physics = {}
params_physics['diffusivity_x'] = state_dict['params']['diffusivity_x']
params_physics['diffusivity_y'] = state_dict['params']['diffusivity_y']
state = create_train_state(params_nn=params_nn, params_physics=params_physics,learning_rate=1e-5)
state = flax.serialization.from_state_dict(state, state_dict)

BATCH_SIZE = 5
ITR_PER_EPOCH = len(FLOW_TIMES)*5

losses = []
min_loss = np.inf
for epoch in range(1000,1500):
    epoch_loss = 0.0
    data_loss = 0.0

    start = time.time()
    hidden_curr_set = []
    cell_curr_set = []
    smoke_curr_set = []
    for set_idx in range(len(SMOKE_FIELD)):
        smoke_curr_set.append(jnp.zeros((ny,nx)))
        hidden_curr_set.append(jnp.zeros((4,128)))
        cell_curr_set.append(jnp.zeros((4,128)))

    time_curr_set = [0 for _ in range(len(SMOKE_FIELD))]

    itr = 0
    while itr < ITR_PER_EPOCH:
        itr_batch_size = min(BATCH_SIZE, ITR_PER_EPOCH-itr)
        state, loss, smoke_curr_set, hidden_curr_set, cell_curr_set, time_curr_set, loss1, loss2, loss3, nsim = \
            train_step(state,
                        batch_size=itr_batch_size,
                        smoke_initial_set=smoke_curr_set,
                        hidden_initial_set=hidden_curr_set,
                        cell_initial_set=cell_curr_set,
                        time_initial_set=time_curr_set,
                        dt=dt,
                        thresh=1e-8)

        itr += BATCH_SIZE
        print(f"EPOCH {epoch}, batch size: {itr_batch_size}, nsim={nsim} loss={loss:.4f}, rel loss={loss1:.4f}, reg loss={loss2}, smooth loss={loss3}")
        epoch_loss += loss
        data_loss += loss1
    # saving every 100 epochs
    if epoch%100==99:
        state_dict = flax.serialization.to_state_dict(state)
        with open(f'{FILENAME_PREFIX}_epoch_{epoch}.pickle', 'wb') as handle:
            pickle.dump(state_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if epoch_loss < min_loss:
        state_dict = flax.serialization.to_state_dict(state)
        with open(f'{FILENAME_PREFIX}_min.pickle', 'wb') as handle:
            pickle.dump(state_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        min_loss = epoch_loss
        print("New minimum achieved")

    losses.append(epoch_loss)
    end = time.time()
    print(f"EPOCH {epoch} (time={(end-start):.1f}): loss={epoch_loss:.4f}, data_loss={data_loss:.4f}")

jax.clear_caches()

### Phase 1.4 Training
# 500 epochs
# first_larger_index = [jnp.searchsorted(dataset['FLOW_TIMES'][i], 300.0, side='right') for i in range(len(dataset['FLOW_TIMES']))]
# FLOW_TIMES = [jnp.array(dataset['FLOW_TIMES'][i][:first_larger_index[i]]) for i in train_set]
# SMOKE_FIELD = [jnp.array(dataset['SMOKE_FIELD'][i][:first_larger_index[i]]) for i in train_set]
FLOW_TIMES = [jnp.array(dataset['FLOW_TIMES'][i]) for i in train_set]
SMOKE_FIELD = [jnp.array(dataset['SMOKE_FIELD'][i]) for i in train_set]

with open(f'{FILENAME_PREFIX}_epoch_1499.pickle', 'rb') as bunch:
  state_dict = pickle.load(bunch)
params_nn = {}
params_nn['closure'] = state_dict['params']['closure']
params_nn['inflow'] = state_dict['params']['inflow']
params_physics = {}
params_physics['diffusivity_x'] = state_dict['params']['diffusivity_x']
params_physics['diffusivity_y'] = state_dict['params']['diffusivity_y']
state = create_train_state(params_nn=params_nn, params_physics=params_physics,learning_rate=1e-5)

BATCH_SIZE = 5
ITR_PER_EPOCH = len(FLOW_TIMES)*10

losses = []
min_loss = np.inf
for epoch in range(1500,2000):
    epoch_loss = 0.0
    data_loss = 0.0

    start = time.time()
    hidden_curr_set = []
    cell_curr_set = []
    smoke_curr_set = []
    for set_idx in range(len(SMOKE_FIELD)):
        smoke_curr_set.append(jnp.zeros((ny,nx)))
        hidden_curr_set.append(jnp.zeros((4,128)))
        cell_curr_set.append(jnp.zeros((4,128)))

    time_curr_set = [0 for _ in range(len(SMOKE_FIELD))]

    itr = 0
    while itr < ITR_PER_EPOCH:
        itr_batch_size = min(BATCH_SIZE, ITR_PER_EPOCH-itr)
        state, loss, smoke_curr_set, hidden_curr_set, cell_curr_set, time_curr_set, loss1, loss2, loss3, nsim = \
            train_step(state,
                        batch_size=itr_batch_size,
                        smoke_initial_set=smoke_curr_set,
                        hidden_initial_set=hidden_curr_set,
                        cell_initial_set=cell_curr_set,
                        time_initial_set=time_curr_set,
                        dt=dt,
                        thresh=1e-8)

        itr += BATCH_SIZE
        print(f"EPOCH {epoch}, batch size: {itr_batch_size}, nsim={nsim} loss={loss:.4f}, rel loss={loss1:.4f}, reg loss={loss2}, smooth loss={loss3}")
        epoch_loss += loss
        data_loss += loss1
    # saving every 100 epochs
    if epoch%100==99:
        state_dict = flax.serialization.to_state_dict(state)
        with open(f'{FILENAME_PREFIX}_epoch_{epoch}.pickle', 'wb') as handle:
            pickle.dump(state_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if epoch_loss < min_loss:
        state_dict = flax.serialization.to_state_dict(state)
        with open(f'{FILENAME_PREFIX}_min.pickle', 'wb') as handle:
            pickle.dump(state_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        min_loss = epoch_loss
        print("New minimum achieved")

    losses.append(epoch_loss)
    end = time.time()
    print(f"EPOCH {epoch} (time={(end-start):.1f}): loss={epoch_loss:.4f}, data_loss={data_loss:.4f}")

jax.clear_caches()