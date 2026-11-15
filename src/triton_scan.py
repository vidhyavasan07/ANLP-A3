import torch
import triton
import triton.language as tl

BLOCK_D = 32
NUM_WARPS = 2


@triton.jit
def _scan_fwd_kernel(
    x_ptr, dt_ptr, a_ptr, b_ptr, c_ptr, d_ptr, y_ptr, h_ptr,
    seq_len, d_inner, d_state,
    STORE_H: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    mask_d = offs_d < d_inner
    mask_n = offs_n < d_state
    mask_dn = mask_d[:, None] & mask_n[None, :]
    a = tl.load(a_ptr + offs_d[:, None] * d_state + offs_n[None, :], mask=mask_dn, other=0.0)
    skip = tl.load(d_ptr + offs_d, mask=mask_d, other=0.0)
    row = pid_b * seq_len
    h = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
    for t in range(0, seq_len):
        x = tl.load(x_ptr + (row + t) * d_inner + offs_d, mask=mask_d, other=0.0)
        dt = tl.load(dt_ptr + (row + t) * d_inner + offs_d, mask=mask_d, other=0.0)
        bt = tl.load(b_ptr + (row + t) * d_state + offs_n, mask=mask_n, other=0.0)
        ct = tl.load(c_ptr + (row + t) * d_state + offs_n, mask=mask_n, other=0.0)
        h = tl.exp(dt[:, None] * a) * h + (dt * x)[:, None] * bt[None, :]
        y = tl.sum(h * ct[None, :], axis=1) + skip * x
        tl.store(y_ptr + (row + t) * d_inner + offs_d, y, mask=mask_d)
        if STORE_H:
            tl.store(
                h_ptr + ((row + t) * d_inner + offs_d[:, None]) * d_state + offs_n[None, :],
                h,
                mask=mask_dn,
            )


@triton.jit
def _scan_bwd_kernel(
    x_ptr, dt_ptr, a_ptr, b_ptr, c_ptr, d_ptr, h_ptr, dy_ptr,
    dx_ptr, ddt_ptr, da_ptr, db_ptr, dc_ptr, dd_ptr,
    seq_len, d_inner, d_state,
    BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    mask_d = offs_d < d_inner
    mask_n = offs_n < d_state
    mask_dn = mask_d[:, None] & mask_n[None, :]
    a = tl.load(a_ptr + offs_d[:, None] * d_state + offs_n[None, :], mask=mask_dn, other=0.0)
    skip = tl.load(d_ptr + offs_d, mask=mask_d, other=0.0)
    row = pid_b * seq_len
    h_cur = tl.load(
        h_ptr + ((row + seq_len - 1) * d_inner + offs_d[:, None]) * d_state + offs_n[None, :],
        mask=mask_dn,
        other=0.0,
    )
    gh = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
    da_acc = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
    dd_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for i in range(0, seq_len):
        t = seq_len - 1 - i
        x = tl.load(x_ptr + (row + t) * d_inner + offs_d, mask=mask_d, other=0.0)
        dt = tl.load(dt_ptr + (row + t) * d_inner + offs_d, mask=mask_d, other=0.0)
        dy = tl.load(dy_ptr + (row + t) * d_inner + offs_d, mask=mask_d, other=0.0)
        bt = tl.load(b_ptr + (row + t) * d_state + offs_n, mask=mask_n, other=0.0)
        ct = tl.load(c_ptr + (row + t) * d_state + offs_n, mask=mask_n, other=0.0)
        h_prev = tl.load(
            h_ptr + ((row + t - 1) * d_inner + offs_d[:, None]) * d_state + offs_n[None, :],
            mask=mask_dn & (t > 0),
            other=0.0,
        )
        decay = tl.exp(dt[:, None] * a)
        gh += dy[:, None] * ct[None, :]
        u = dt * x
        dc_t = tl.sum(dy[:, None] * h_cur, axis=0)
        db_t = tl.sum(gh * u[:, None], axis=0)
        du = tl.sum(gh * bt[None, :], axis=1)
        da_decay = gh * h_prev * decay
        ddt = du * x + tl.sum(da_decay * a, axis=1)
        dx = du * dt + dy * skip
        da_acc += da_decay * dt[:, None]
        dd_acc += dy * x
        tl.store(dx_ptr + (row + t) * d_inner + offs_d, dx, mask=mask_d)
        tl.store(ddt_ptr + (row + t) * d_inner + offs_d, ddt, mask=mask_d)
        tl.atomic_add(dc_ptr + (row + t) * d_state + offs_n, dc_t, mask=mask_n)
        tl.atomic_add(db_ptr + (row + t) * d_state + offs_n, db_t, mask=mask_n)
        gh = gh * decay
        h_cur = h_prev
    tl.atomic_add(da_ptr + offs_d[:, None] * d_state + offs_n[None, :], da_acc, mask=mask_dn)
    tl.atomic_add(dd_ptr + offs_d, dd_acc, mask=mask_d)


def _launch_forward(x, dt, a, b, c, d, store_h):
    batch, seq_len, d_inner = x.shape
    d_state = a.shape[1]
    y = torch.empty_like(x)
    if store_h:
        h = torch.empty((batch, seq_len, d_inner, d_state), device=x.device, dtype=torch.float32)
    else:
        h = y
    grid = (batch, triton.cdiv(d_inner, BLOCK_D))
    _scan_fwd_kernel[grid](
        x, dt, a, b, c, d, y, h, seq_len, d_inner, d_state,
        STORE_H=store_h,
        BLOCK_D=BLOCK_D,
        BLOCK_N=triton.next_power_of_2(d_state),
        num_warps=NUM_WARPS,
    )
    return y, h


class SelectiveScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dt, a, b, c, d):
        x, dt, a, b, c, d = (t.contiguous() for t in (x, dt, a, b, c, d))
        y, h = _launch_forward(x, dt, a, b, c, d, True)
        ctx.save_for_backward(x, dt, a, b, c, d, h)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, dt, a, b, c, d, h = ctx.saved_tensors
        dy = dy.contiguous()
        batch, seq_len, d_inner = x.shape
        d_state = a.shape[1]
        dx = torch.empty_like(x)
        ddt = torch.empty_like(dt)
        da = torch.zeros_like(a)
        db = torch.zeros_like(b)
        dc = torch.zeros_like(c)
        dd = torch.zeros_like(d)
        grid = (batch, triton.cdiv(d_inner, BLOCK_D))
        _scan_bwd_kernel[grid](
            x, dt, a, b, c, d, h, dy,
            dx, ddt, da, db, dc, dd,
            seq_len, d_inner, d_state,
            BLOCK_D=BLOCK_D,
            BLOCK_N=triton.next_power_of_2(d_state),
            num_warps=NUM_WARPS,
        )
        return dx, ddt, da, db, dc, dd


def selective_scan(x, dt, a, b, c, d):
    needs_grad = torch.is_grad_enabled() and any(t.requires_grad for t in (x, dt, a, b, c, d))
    if needs_grad:
        return SelectiveScan.apply(x, dt, a, b, c, d)
    x, dt, a, b, c, d = (t.contiguous() for t in (x, dt, a, b, c, d))
    return _launch_forward(x, dt, a, b, c, d, False)[0]