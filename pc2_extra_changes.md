# Additional Changes in tdmpc_square_public Beyond Documentation

This document identifies significant changes in `tdmpc_square_public` that go beyond what's mentioned in the paper/README. These are changes that may contribute to the reported performance gains but weren't explicitly documented.

---

## 1. **Seed Steps Calculation** [CRITICAL - Likely Major Impact]

This is the most significant undocumented difference you correctly identified.

### tdmpc_square_public (`envs/__init__.py:91`):
```python
cfg.seed_steps = max(1000, 5 * cfg.episode_length)
```
For `humanoid-walk` with `episode_length=250`: **seed_steps = max(1000, 1250) = 1250 steps**

### newt (`trainer.py:326` + `config.py:59`):
```python
seeding_coef: int = 5
# In trainer:
if self._step >= self.cfg.seeding_coef * self._update_freq:
# where _update_freq = num_envs * episode_length * world_size
```
For `humanoid-walk` with `num_envs=20`, `episode_length=250`, `world_size=1`:
**seed_steps = 5 * 20 * 250 * 1 = 25,000 steps**

### Impact Analysis

| Setting | tdmpc_square | newt | Ratio |
|---------|--------------|------|-------|
| seed_steps | 1,250 | 25,000 | **20x more** |

**Why this matters**:
- tdmpc_square starts training after just 5 episodes (1250 steps / 250 ep_length)
- newt starts training after 100 episodes (25000 steps / 250 ep_length)
- This means newt spends much more time collecting random data before policy learning begins
- tdmpc_square gets policy updates earlier, allowing the planner to guide learning sooner

**Fix**: Modify your seeding logic to match:
```python
# In config.py or envs/__init__.py
cfg.seed_steps = max(1000, 5 * cfg.episode_length)
# Then in trainer.py:
if self._step >= cfg.seed_steps:
```

---

## 2. **Multiple Actor Modes** [MODERATE]

tdmpc_square supports 5 different actor training modes via `actor_mode` config:

### tdmpc_square_public (`tdmpc_square.py:344-394`):
```python
if self.cfg.actor_mode=="sac":
    # TD-MPC2 baseline setting
    pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()

elif self.cfg.actor_mode=="awac":
    # AWAC-MPC with advantage weighting
    with torch.no_grad():
        vs = self.model.Q(zs, action, task, return_type="avg")
        vs = self.scale(vs)
    adv = (qs - vs).detach()
    weights = torch.clamp(torch.exp(adv / self.cfg.awac_lambda), cfg.exp_adv_min, cfg.exp_adv_max)
    log_pis_action = self.model.log_pi_action(zs, action, task)
    pi_loss = (( - weights * log_pis_action).mean(dim=(1, 2)) * rho).mean()

elif self.cfg.actor_mode=="residual":
    # TD-M(PC)^2 mode (the one you implemented)
    ...

elif self.cfg.actor_mode=="bc_sac":
    # BC-SAC hybrid
    q_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()
    prior_loss = (((pis - action) ** 2).sum(dim=-1).mean(dim=1) * rho).mean()
    pi_loss = q_loss + self.cfg.prior_coef * prior_loss

elif self.cfg.actor_mode=="bc":
    # Pure behavior cloning from planner
    log_pis_prior = math.gaussian_logprob(eps, std.log(), size=action_dims).mean(dim=-1)
    pi_loss = - (log_pis_prior.mean(dim=-1) * rho).mean()
```

**Your implementation**: Only has 3 modes (0: disabled, 1: alternate PC/BC, 2: TD-M(PC)^2)

**Note**: The "residual" mode is what you want. AWAC mode might be worth trying too.

---

## 3. **log_pi_action Method for AWAC** [MINOR]

### tdmpc_square_public (`common/world_model.py:171-189`):
```python
def log_pi_action(self, z, a, task):
    """
    Compute the log probability of an action sequence given the latent states.
    """
    if self.cfg.multitask:
        z = self.task_emb(z, task)
    mu, log_std = self._pi(z).chunk(2, dim=-1)
    eps = (a - mu) / (log_std.exp() + 1e-8)
    ...
    log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
    return log_pi
```

**newt**: Does not have this method. Only needed if you implement AWAC mode.

---

## 4. **Pretraining on Seed Data** [MODERATE]

### tdmpc_square_public (`trainer/online_trainer.py:182-190`):
```python
if self._step >= self.cfg.seed_steps:
    if self._step == self.cfg.seed_steps:
        num_updates = self.cfg.seed_steps  # All seed data at once!
        print("Pretraining agent on seed data...")
    else:
        num_updates = 1
    for _ in range(num_updates):
        _train_metrics = self.agent.update(self.buffer)
```

When training starts, tdmpc_square does `seed_steps` gradient updates at once (1250 updates for humanoid-walk).

### newt (`trainer.py:326-333`):
```python
if self._step >= self.cfg.seeding_coef * self._update_freq:
    self._update_tokens += self.cfg.num_envs * self.cfg.world_size * self.cfg.utd
    if self._update_tokens >= 1.0:
        num_updates = int(self._update_tokens)
        for _ in range(num_updates):
            _train_metrics = self.agent.update(self.buffer)
```

**newt uses a token-based UTD (Update-to-Data) ratio** rather than a big pretraining burst.

**Impact**: tdmpc_square gets a large initial training boost from seed data before continuing with online learning.

---

## 5. **Evaluation Modes** [MINOR]

### tdmpc_square_public (`config.yaml:11-13`):
```yaml
eval_pi: true        # Evaluate nominal policy (without MPC)
eval_value: true     # Evaluate value function approximation
```

### tdmpc_square_public (`trainer/online_trainer.py:49-67`):
```python
if self.cfg.eval_pi:
    # Evaluate nominal policy pi (without planner)
    for i in range(self.cfg.eval_episodes):
        ...
        action, _, _ = self.agent.act(obs, t0=t == 0, eval_mode=True, use_pi=True)
        ...
```

**Impact**: tdmpc_square tracks both MPC performance and pure policy performance. This helps verify that the policy is actually learning from the planner.

---

## 6. **Mu/Std Storage During Seeding** [MODERATE]

### tdmpc_square_public (`trainer/online_trainer.py:174-176`):
```python
if self._step > self.cfg.seed_steps:
    t0 = len(self._tds) == 1
    action, mu, std = self.agent.act(obs, t0=t0)
else:
    action = self.env.rand_act()
    mu, std = action.detach().clone(), torch.full_like(action, math.exp(self.cfg.log_std_max))
```

During random seeding:
- `mu = action` (the random action itself)
- `std = exp(log_std_max) = exp(2) = 7.389`

### newt (`trainer.py:265-271`):
```python
use_mpc = True  # Always True!
use_agent = True
if use_agent:
    torch.compiler.cudagraph_mark_step_begin()
    action, mu, std = self.agent(obs, t0=done, step=self._step, task=self._tasks, mpc=use_mpc)
else:
    action = self.env.rand_act()
```

**Difference**: Your newt code always uses the agent even during what should be "seeding". The commented-out code shows the intended behavior but it's not active.

---

## 7. **Q-Network Ensemble Implementation** [MINOR]

### tdmpc_square_public (`common/layers.py:7-26`):
```python
class Ensemble(nn.Module):
    def __init__(self, modules, **kwargs):
        modules = nn.ModuleList(modules)
        fn, params, _ = combine_state_for_ensemble(modules)  # Uses functorch
        self.vmap = torch.vmap(fn, in_dims=(0, 0, None), randomness="different")
        self.params = nn.ParameterList([nn.Parameter(p) for p in params])
```

### newt (`common/layers.py:10-35`):
```python
class Ensemble(nn.Module):
    def __init__(self, modules, **kwargs):
        self.params = from_modules(*modules, as_module=True)  # Uses tensordict
        with self.params[0].data.to("meta").to_module(modules[0]):
            self.module = deepcopy(modules[0])
```

**Impact**: Different vmap implementations. Both achieve the same goal but tensordict's `from_modules` might have different graph compatibility with torch.compile.

---

## 8. **Action Dimension Normalization** [MODERATE]

Both implementations normalize the prior loss by action dimension:

```python
pi_loss = q_loss + (self.cfg.prior_coef * self.cfg.action_dim / 61) * prior_loss
```

**Why 61?** This is the action dimension of the DMControl Dog tasks. The paper likely tuned `prior_coef` on dog tasks and this normalization keeps the loss scale consistent across tasks.

For `humanoid-walk` with `action_dim=21`: factor = `21/61 = 0.344`

---

## 9. **AWAC-Specific Config Parameters** [MINOR]

### tdmpc_square_public (`config.yaml:51-53`):
```yaml
awac_lambda: 0.3333
exp_adv_min: 0.1
exp_adv_max: 10.0
```

These are used for AWAC mode advantage weighting.

---

## Summary: Key Undocumented Changes That May Explain Performance Gap

1. **Seed steps**: 20x smaller seeding period in tdmpc_square (1250 vs 25000 steps)
2. **Pretraining burst**: Large initial training batch when seeding completes
3. **Mu/std during seeding**: Proper storage of planner statistics even during random exploration
4. **prior_coef**: 1.0 vs your 10.0 (already in discrepancies.md)

---

## Recommended Actions

### High Priority:
1. **Fix seed_steps calculation**:
   ```python
   # In envs/__init__.py or equivalent:
   cfg.seed_steps = max(1000, 5 * cfg.episode_length)
   ```

2. **Add pretraining burst**:
   ```python
   if self._step == cfg.seed_steps:
       num_updates = cfg.seed_steps
       print("Pretraining agent on seed data...")
   ```

3. **Fix seeding data collection** to use random actions with proper mu/std storage:
   ```python
   if self._step <= cfg.seed_steps:
       action = self.env.rand_act()
       mu = action.clone()
       std = torch.full_like(action, math.exp(cfg.log_std_max))
   ```

### Medium Priority:
4. **Add `eval_pi` mode** to verify policy is learning from planner
5. **Consider trying AWAC mode** for comparison

### Low Priority:
6. **Add dropout** to Q-networks
7. **Implement remaining actor modes** for ablation studies
