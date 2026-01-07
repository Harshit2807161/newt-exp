# Bugs and Issues in tdmpc_square_public

This document lists all bugs, missing dependencies, and issues that prevent `tdmpc_square_public` from running successfully.

---

## 1. Missing Dependencies in requirements_tdmpc_square.txt [CRITICAL]

The requirements file is incomplete. Missing packages:

```
# Missing from requirements_tdmpc_square.txt:
dm-control          # Required for dmcontrol.py
dm_env              # Required for dmcontrol.py (import dm_env)
mujoco              # Required for dm_control
gymnasium           # Required but version not specified
numpy               # Required but version not specified
imageio             # Required for evaluate.py video saving
humanoid_bench      # Required for humanoid.py (if using humanoid tasks)
functorch           # Required for common/layers.py (from functorch import combine_state_for_ensemble)
```

**Current requirements_tdmpc_square.txt only has:**
```
--editable tdmpc_square
torch==2.3.1
torchaudio==2.3.1
torchrl==0.4.0
torchvision==0.18.1
hydra-core==1.3.2
hydra-submitit-launcher==1.2.0
pyquaternion==0.9.9
tensordict==0.4.0
pandas==2.2.2
termcolor==2.4.0
wandb
```

**Fix**: Add the missing dependencies:
```bash
pip install dm-control mujoco gymnasium numpy imageio
# For functorch, it's included in PyTorch >= 2.0 as torch.func
```

---

## 2. dm_env Import Error [CRITICAL - Your Current Blocker]

**File**: `tdmpc_square/tdmpc_square/envs/dmcontrol.py:4`

```python
import dm_env  # This fails if dm-control is not installed
```

**Root Cause**: `dm_env` is a dependency of `dm-control` but must be installed separately in some cases.

**Fix**:
```bash
pip install dm-env
# OR install dm-control which should include it:
pip install dm-control
```

**Note**: dm-control requires MuJoCo to be installed. Make sure you have:
```bash
pip install mujoco
```

---

## 3. functorch Import Deprecation [MODERATE]

**File**: `tdmpc_square/tdmpc_square/common/layers.py:4`

```python
from functorch import combine_state_for_ensemble
```

**Issue**: `functorch` is deprecated and integrated into PyTorch 2.0+. The import should be:

```python
from torch.func import stack_module_state  # Replacement for combine_state_for_ensemble
# OR
from torch._functorch.make_functional import make_functional_with_buffers
```

**Impact**: This may work with older PyTorch versions but will break with newer ones.

---

## 4. Missing Model Size in config.yaml [MODERATE]

**File**: `tdmpc_square/tdmpc_square/config.yaml:61`

```yaml
model_size: ???
```

**Issue**: `model_size` is set to `???` (required) but must be provided to run. If not specified, you'll get a Hydra error.

**Fix**: Either provide at command line:
```bash
python train.py model_size=5
```

Or set a default in config.yaml:
```yaml
model_size: 5  # Instead of ???
```

---

## 5. Missing checkpoint Path for Training [MINOR]

**File**: `tdmpc_square/tdmpc_square/config.yaml:9`

```yaml
checkpoint: ???
```

**Issue**: `checkpoint` is required (`???`) but only needed for evaluation/loading. For fresh training, this causes errors.

**Fix**: Change to optional:
```yaml
checkpoint: null  # Instead of ???
```

---

## 6. Missing data_dir Path [MODERATE]

**File**: `tdmpc_square/tdmpc_square/config.yaml:30`

```yaml
data_dir: ???
```

**Issue**: Required but only used for offline/multitask training. Causes errors even for online single-task.

**Fix**: Change to optional or provide a default:
```yaml
data_dir: ./data  # Instead of ???
```

---

## 7. Humanoid Environment Task Name Format [MODERATE]

**File**: `tdmpc_square/tdmpc_square/envs/humanoid.py:42-43`

```python
def make_env(cfg):
    if not cfg.task.startswith("humanoid_"):
        raise ValueError("Unknown task:", cfg.task)
```

**Issue**: The check uses `humanoid_` (underscore) but config.yaml has:
```yaml
task: humanoid-walk  # Uses hyphen, not underscore
```

**The code also does**:
```python
cfg.task.removeprefix("humanoid_")  # Expects underscore
```

**Fix**: Either use consistent naming or fix the check:
```python
if not cfg.task.startswith("humanoid-"):  # Use hyphen
    raise ValueError("Unknown task:", cfg.task)
# ...
env = gym.make(cfg.task.removeprefix("humanoid-"), ...)  # Use hyphen
```

---

## 8. MultitaskWrapper API Incompatibility [MODERATE]

**File**: `tdmpc_square/tdmpc_square/envs/wrappers/multitask.py:58-68`

```python
def reset(self, task_idx=-1):
    # ...
    return self._pad_obs(self.env.reset())  # Returns 1 value

def step(self, action):
    obs, reward, done, info = self.env.step(...)  # Expects 4 values
    return self._pad_obs(obs), reward, done, info  # Returns 4 values
```

**Issue**:
1. `reset()` expects gym's old API (returns just obs) but Gymnasium returns `(obs, info)`
2. `step()` expects old API (4 values) but Gymnasium returns 5 values `(obs, reward, terminated, truncated, info)`

**Fix**:
```python
def reset(self, task_idx=-1):
    # ...
    obs, info = self.env.reset()
    return self._pad_obs(obs), info

def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(...)
    return self._pad_obs(obs), reward, terminated, truncated, info
```

---

## 9. TensorWrapper Uses Old Gym API in Some Places [MINOR]

**File**: `tdmpc_square/tdmpc_square/envs/wrappers/tensor.py:33-35`

```python
def reset(self, task_idx=None):
    obs, info = self.env.reset()
    return self._obs_to_tensor(obs), info  # Correct!
```

This is correct, but inconsistent with MultitaskWrapper.

---

## 10. evaluate.py Uses Old Gym API [MODERATE]

**File**: `tdmpc_square/tdmpc_square/evaluate.py:103-108`

```python
obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0  # Wrong: reset() returns (obs, info)
# ...
obs, reward, done, info = env.step(action)  # Wrong: step() returns 5 values
```

**Fix**:
```python
(obs, _), done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0
# ...
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

---

## 11. online_trainer.py success_subtasks Key Error [MINOR]

**File**: `tdmpc_square/tdmpc_square/trainer/online_trainer.py:160-161`

```python
results_metrics = {'return': train_metrics['episode_reward'],
                   'episode_length': len(self._tds[1:]),
                   'success': train_metrics['episode_success'],
                   'success_subtasks': info['success_subtasks'],  # May not exist!
                   'step': self._step,}
```

**Issue**: `info['success_subtasks']` may not exist in all environments, causing KeyError.

**Fix**:
```python
'success_subtasks': info.get('success_subtasks', 0),
```

---

## 12. Missing ??? Values in config.yaml [CRITICAL]

Multiple config values are set to `???` (required in Hydra) but don't have defaults:

```yaml
checkpoint: ???
data_dir: ???
model_size: ???
work_dir: ???
task_title: ???
multitask: ???
tasks: ???
obs_shape: ???
action_dim: ???
episode_length: ???
obs_shapes: ???
action_dims: ???
episode_lengths: ???
seed_steps: ???
bin_size: ???
policy_path: ???
mean_path: ???
var_path: ???
policy_type: ???
small_obs: ???
```

**Issue**: Many of these are computed at runtime by `parse_cfg()` or `make_env()`, but Hydra will fail if they're accessed before being set.

**Fix**: Set computed values to `null` instead of `???`:
```yaml
work_dir: null
task_title: null
multitask: null
# etc.
```

---

## 13. Empty core_requirements in setup.py [MINOR]

**File**: `tdmpc_square/setup.py:7-8`

```python
core_requirements = [
]
```

**Issue**: No dependencies are specified, so `pip install -e tdmpc_square` won't install anything.

---

## 14. TimeStepToGymWrapper Missing truncated Return [MINOR]

**File**: `tdmpc_square/tdmpc_square/envs/dmcontrol.py:173-182`

```python
def step(self, action):
    self.t += 1
    time_step = self.env.step(action)
    return (
        self._obs_to_array(time_step.observation),
        time_step.reward,
        time_step.last(),
        self.t == self.max_episode_steps,  # This is truncated, but position is wrong
        defaultdict(float),
    )
```

**Issue**: Returns 5 values but the order might be confusing. The 4th value should be `truncated` but it's checking time limit while `time_step.last()` combines both.

---

## 15. RunningScale.value Returns Float vs Tensor [MINOR]

**File**: `tdmpc_square/tdmpc_square/common/scale.py`

This file wasn't shown but based on the main agent code:
```python
if self.scale.value > self.cfg.scale_threshold
```

The comparison works because `.value` returns a Python float. But if your implementation returns a tensor, you need `.item()`.

---

## Quick Fix Script

Create a file `fix_dependencies.sh`:

```bash
#!/bin/bash
# Install missing dependencies for tdmpc_square_public

pip install dm-control dm-env mujoco gymnasium numpy imageio

# If you need humanoid_bench:
# pip install humanoid-bench

# Verify installation
python -c "import dm_env; print('dm_env OK')"
python -c "from dm_control import suite; print('dm_control OK')"
python -c "import gymnasium; print('gymnasium OK')"
```

---

## Recommended Running Command

After fixing dependencies, run with explicit config:

```bash
cd tdmpc_square_public/tdmpc_square

python -m tdmpc_square.train \
    task=walker-walk \
    model_size=5 \
    checkpoint=null \
    data_dir=./data \
    steps=1000000 \
    disable_wandb=true
```

Or for DMControl tasks:
```bash
python -m tdmpc_square.train \
    task=cheetah-run \
    model_size=5 \
    checkpoint=null \
    steps=1000000
```

---

## Summary of Critical Blockers

| Priority | Issue | Fix |
|----------|-------|-----|
| CRITICAL | Missing dm_env/dm-control | `pip install dm-control dm-env mujoco` |
| CRITICAL | Missing `???` config values | Change to `null` or provide defaults |
| CRITICAL | functorch deprecation | Use `torch.func` instead |
| HIGH | Gym API incompatibility | Update wrappers to Gymnasium 5-tuple API |
| HIGH | humanoid task name format | Use consistent hyphen/underscore |
| MEDIUM | Empty setup.py requirements | Add dependencies |
| LOW | success_subtasks KeyError | Use `.get()` with default |
