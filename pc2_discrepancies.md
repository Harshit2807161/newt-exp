# TD-M(PC)^2 Implementation Discrepancies Analysis

This document identifies discrepancies between your implementation in `newt/tdmpc2` (constrained-planning branch) and the reference implementation in `tdmpc_square_public/tdmpc_square`.

---

## 1. **prior_coef Default Value** [CRITICAL]

| Aspect | tdmpc_square_public | newt |
|--------|---------------------|------|
| Default value | `prior_coef: 1.0` | `prior_coef: 10.0` |
| Location | `config.yaml:48` | `config.py:44` |

**Impact**: A 10x larger prior coefficient may overpower the Q-loss term, causing the policy to become too conservative or focus too heavily on imitating the planner distribution rather than maximizing returns.

**Fix**: Change `prior_coef` to `1.0` in your config.

---

## 2. **gaussian_logprob Function Implementation** [CRITICAL]

### tdmpc_square_public (`common/math.py:27-32`):
```python
def gaussian_logprob(eps, log_std, size=None):
    """Compute Gaussian log probability."""
    residual = _gaussian_residual(eps, log_std).sum(-1, keepdim=True)
    if size is None:
        size = eps.size(-1)
    return _gaussian_logprob(residual) * size
```

### newt (`common/math.py:28-39`):
```python
def gaussian_logprob_constrained(eps, log_std, size=None):
    """Compute Gaussian log probability."""
    residual = _gaussian_residual(eps, log_std).sum(-1, keepdim=True)
    if size is None:
        size = eps.size(-1)
    return _gaussian_logprob(residual) * size

def gaussian_logprob(eps, log_std):
    """Compute Gaussian log probability."""
    residual = -0.5 * eps.pow(2) - log_std
    log_prob = residual - 0.9189385175704956  # 0.5 * log(2*pi)
    return log_prob.sum(-1, keepdim=True)
```

**Discrepancy**:
- tdmpc_square uses `gaussian_logprob(..., size=action_dims)` which multiplies the result by `action_dims`
- newt's `gaussian_logprob_constrained` is equivalent but uses it differently in context

**The key difference is in the scaling**: In tdmpc_square, the function explicitly multiplies by `size` (action_dims for multitask). Your `_constrained_pi_loss` uses `gaussian_logprob_constrained` which is correct, but verify the `.mean(dim=-1)` behavior matches.

---

## 3. **RunningScale.value Property** [MODERATE]

### tdmpc_square_public (`common/scale.py:25-27`):
```python
@property
def value(self):
    return self._value.cpu().item()  # Returns a Python float
```

### newt (`common/scale.py:10`):
```python
self.register_buffer('value', torch.ones(1, ...))  # Returns a tensor
```

**Impact**: When used in comparisons like `self.scale.value > self.cfg.scale_threshold`, tdmpc_square compares floats while newt compares tensors. This works but may have subtle numerical differences.

---

## 4. **Policy Network Output Format** [CRITICAL]

### tdmpc_square_public (`common/world_model.py:143-169`):
```python
def pi(self, z, task):
    ...
    return mu, pi, log_pi, log_std  # Returns 4 values
```

### newt (`common/world_model.py:156-200`):
```python
def pi(self, z, task):
    ...
    info = TensorDict({
        "mean": mean,
        "log_std": log_std,
        "entropy": -log_prob,
        "scaled_entropy": -log_prob * entropy_scale,
    })
    return action, info  # Returns 2 values (action, info dict)
```

**Impact**: The `update_pi` functions access these differently:
- tdmpc_square: `_, pis, log_pis, _ = self.model.pi(zs, task)`
- newt: `pi_action, info = self.model.pi(zs, task)`; `log_pis = -info["entropy"]`

This is fine as long as the mapping is correct. **Verify that `-info["entropy"]` equals `log_pis` from tdmpc_square**.

---

## 5. **Policy Loss Computation - Key Equation Differences** [CRITICAL]

### tdmpc_square_public (`tdmpc_square.py:362-374`):
```python
elif self.cfg.actor_mode=="residual":
    # Loss for TD-M(PC)^2
    action_dims = None if not self.cfg.multitask else self.model._action_masks.size(-1)
    std = torch.max(std, self.cfg.min_std * torch.ones_like(std))
    eps = (pis - mu) / std
    log_pis_prior = math.gaussian_logprob(eps, std.log(), size=action_dims).mean(dim=-1)

    log_pis_prior = self.scale(log_pis_prior) if self.scale.value > self.cfg.scale_threshold else torch.zeros_like(log_pis_prior)

    q_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()
    prior_loss = - (log_pis_prior.mean(dim=-1) * rho).mean()
    pi_loss = q_loss + (self.cfg.prior_coef * self.cfg.action_dim / 61) * prior_loss
```

### newt (`tdmpc2.py:297-316`):
```python
def _constrained_pi_loss(self, zs, action, mu, std, task):
    ...
    std = torch.max(std, self.cfg.min_std * torch.ones_like(std))
    eps = (pis - mu) / std
    log_pis_prior = math.gaussian_logprob_constrained(eps, std.log()).mean(dim=-1)

    log_pis_prior = self.scale(log_pis_prior) if self.scale.value > self.cfg.scale_threshold else torch.zeros_like(log_pis_prior)

    q_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * self.rho[:-1]).mean()
    prior_loss = - (log_pis_prior.mean(dim=-1) * self.rho[:-1]).mean()
    pi_loss = q_loss + (self.cfg.prior_coef * self.cfg.action_dim / 61) * prior_loss
```

**Key Differences**:
1. **rho indexing**: tdmpc_square uses full `rho`, newt uses `self.rho[:-1]` (excluding last timestep)
2. **gaussian_logprob**: tdmpc_square passes `size=action_dims` for multitask, newt doesn't pass size explicitly
3. Both use the same action dimension normalization factor (`/ 61`, which is the dog task's action dim)

---

## 6. **Rho (Temporal Weighting) Computation** [MODERATE]

### tdmpc_square_public (`tdmpc_square.py:343`):
```python
rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
# NOT normalized
```

### newt (`tdmpc2.py:50-51`):
```python
self.rho = torch.pow(self.cfg.rho, torch.arange(self.cfg.horizon+1, device=self.device))
self.rho = self.rho / self.rho.sum()  # NORMALIZED!
```

**Impact**: Newt normalizes rho so it sums to 1, while tdmpc_square does not. This affects the scale of the loss terms. With `rho=0.5` and `horizon=3`:
- tdmpc_square: `[1, 0.5, 0.25, 0.125]` (sum = 1.875)
- newt: `[0.533, 0.267, 0.133, 0.067]` (sum = 1.0)

**This is a significant scaling difference that affects the magnitude of losses!**

---

## 7. **Network Architecture Sizes** [CRITICAL - Performance Impact]

| Parameter | tdmpc_square_public | newt |
|-----------|---------------------|------|
| `num_enc_layers` | 2 | 3 |
| `enc_dim` | 256 | 1024 |
| `mlp_dim` | 512 | 1024 |
| `task_dim` | 96 | 512 |

**Impact**: Newt uses significantly larger networks (4x wider encoder, 2x wider MLP). This:
- Increases computational cost significantly
- May require different learning rates or hyperparameters
- Could lead to overfitting with smaller datasets

**Your comment in config.py notes this**: `# DEFAULT 512 IN TDM(PC)^2`

---

## 8. **Dropout in Q-Networks** [MINOR]

### tdmpc_square_public (`common/layers.py:41-48`):
```python
layers.mlp(..., dropout=cfg.dropout)  # cfg.dropout = 0.01
```

### newt (`common/layers.py:213-225`):
```python
def mlp(in_dim, mlp_dims, out_dim, act=None):
    # No dropout parameter!
```

**Impact**: tdmpc_square uses 1% dropout in Q-networks for regularization. Newt has no dropout.

---

## 9. **Scale Update Order** [MINOR]

### tdmpc_square_public (`tdmpc_square.py:339-341`):
```python
self.scale.update(qs[0])  # Update BEFORE scaling
qs = self.scale(qs)       # Then scale
```

### newt (`tdmpc2.py:304-305`):
```python
qs = self.model.Q(zs, pi_action, task, return_type='avg')
qs = self.scale(qs)  # Scale first
# ... then later in update_pi:
self.scale.update(qs)  # Update with unscaled qs
```

**Both are correct** - they both update scale with unscaled Q-values. But verify the update happens in the right place.

---

## Summary of Critical Fixes Needed

1. **Change `prior_coef` from 10.0 to 1.0**
2. **Verify rho normalization** - consider removing the `/ self.rho.sum()` normalization
3. **Consider reducing architecture size** to match tdmpc_square for fair comparison
4. **Add dropout to Q-networks** if you want to match exactly
5. **Verify `gaussian_logprob_constrained` behavior** matches tdmpc_square's `gaussian_logprob(..., size=action_dims)`

---

## Quick Test Commands

To verify your implementation matches, you can:
```python
# Test gaussian_logprob equivalence
eps = torch.randn(3, 256, 21)
log_std = torch.randn(3, 256, 21)
# tdmpc_square style
result1 = math.gaussian_logprob(eps, log_std, size=21).mean(dim=-1)
# newt style
result2 = math.gaussian_logprob_constrained(eps, log_std).mean(dim=-1)
# Should be equal
```
