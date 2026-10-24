def compute_act_loss(
    predicted_actions,
    target_actions,
    action_is_pad,
    mu,
    logvar,
    beta=6,
    free_bits=0.0,
):
    """ACT 损失 = 动作 L1 + beta * KL。

    参数 / Args:
        free_bits: 每维 KL 的下限（nat）。低于该值的维度不产生 KL 梯度，
            避免 KL 项把后验直接压成 N(0,1)。
            每维 KL floor in nats; dims below it get zero KL gradient, which
            stops the KL term from collapsing the posterior to N(0,1).
    """
    errors = (predicted_actions - target_actions).abs()  # [B,16,2]
    errors = errors.mean(dim=-1)  # [B,16]

    valid_mask = ~action_is_pad
    valid_errors = errors.masked_fill(~valid_mask, 0.0)
    # average over valid only
    action_loss = valid_errors.sum() / valid_mask.sum().clamp_min(1)

    # kl between the diagonal gaussian and the standard normal
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B,32]

    # 返回的 kl_loss 始终是未截断的真实 KL，用于日志和监控
    # The reported kl_loss is always the true, unclamped KL
    kl_loss = kl_per_dim.sum(dim=-1).mean()

    # 只有优化目标使用 free-bits 下限；低于下限的维度梯度为零。
    # 注意 free_bits 必须明显小于真实的每维 KL，否则整个 KL 项会失去梯度。
    # Only the objective uses the free-bits floor; dims below it get no gradient.
    # free_bits must stay well below the true per-dim KL, or the whole term goes inert.
    if free_bits > 0:
        kl_objective = kl_per_dim.clamp_min(free_bits).sum(dim=-1).mean()
    else:
        kl_objective = kl_loss

    total_loss = action_loss + beta * kl_objective
    return total_loss, action_loss, kl_loss
