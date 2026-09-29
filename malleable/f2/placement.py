"""The F2 address map: how the accelerator's two 2 GiB channels sit in the PCIS window
and in the HBM pseudo-channels (PCs).

Must match f2/rtl/f2_hbm_adapter.sv and f2/sim/tb_f2_shell.sv. Host software never needs
the PC placement (it addresses channel c at HBM_BASE + CH_BASE[c] + offset, exactly as the
original board's driver addresses BASE[c] + offset); this module exists to state the map
once, to test it, and to plan capacity.
"""
from __future__ import annotations

import numpy as np

HBM_BASE = 1 << 36                      # PCIS address of the HBM window (AWS HDK examples)
WINDOW_BYTES = 1 << 32                  # 4 GiB: two 2 GiB core channels
CH_BASE = (0x0000_0000, 0x8000_0000)    # channel c at CH_BASE[c] inside the window
CH_BYTES = 1 << 31                      # channel address space (2 GiB)
BEAT = 64                               # bytes per core AXI beat
STRIPE_BYTES = 512                      # bytes per stripe (one 16-beat 256-bit burst)
HBM_PC_BYTES = 512 << 20                # one real HBM pseudo-channel (16 GiB / 32)
HBM_PCS = 32


def pcis_address(ch: int, off: int) -> int:
    """PCIS (AppPF BAR4) address of byte `off` of core channel `ch`."""
    if ch not in (0, 1) or not 0 <= off < CH_BYTES:
        raise ValueError("channel or offset out of range")
    return HBM_BASE + CH_BASE[ch] + off


def in_window(addr: int) -> bool:
    return HBM_BASE <= addr < HBM_BASE + WINDOW_BYTES


def place(ch: int, off: int, pcs_per_ch: int = 2, pc_aw: int = 29) -> tuple[int, int]:
    """(global PC index, PC-local byte address) of channel byte `off`. Raises when the
    byte does not fit in the channel's PCs (the RTL answers DECERR)."""
    stripe = off // STRIPE_BYTES
    pc = ch * pcs_per_ch + stripe % pcs_per_ch
    local = (stripe // pcs_per_ch) * STRIPE_BYTES + off % STRIPE_BYTES
    if local >> pc_aw:
        raise ValueError(f"channel {ch} offset {off:#x} is beyond the {pcs_per_ch} PCs of 2**{pc_aw} bytes")
    return pc, local


def place_many(ch: int, offs: np.ndarray, pcs_per_ch: int = 2) -> tuple[np.ndarray, np.ndarray]:
    offs = np.asarray(offs, np.int64)
    stripe = offs // STRIPE_BYTES
    return ch * pcs_per_ch + stripe % pcs_per_ch, (stripe // pcs_per_ch) * STRIPE_BYTES + offs % STRIPE_BYTES


def channel_capacity(pcs_per_ch: int = 2, pc_aw: int = 29) -> int:
    """Bytes of one core channel that fit in its PCs (capped by the 2 GiB address space)."""
    return min(CH_BYTES, pcs_per_ch << pc_aw)


def split_channel(image: np.ndarray, ch: int, pcs_per_ch: int = 2) -> dict[int, np.ndarray]:
    """One channel's bytes -> {global PC: that PC's local bytes} (length multiple of the stripe)."""
    image = np.asarray(image, np.uint8)
    if len(image) % (STRIPE_BYTES * pcs_per_ch):
        raise ValueError("image must be a multiple of pcs_per_ch * 512 bytes")
    v = image.reshape(-1, pcs_per_ch, STRIPE_BYTES)
    return {ch * pcs_per_ch + k: np.ascontiguousarray(v[:, k, :]).reshape(-1) for k in range(pcs_per_ch)}


def join_channel(parts: dict[int, np.ndarray], ch: int, pcs_per_ch: int = 2) -> np.ndarray:
    """Inverse of split_channel."""
    cols = [np.asarray(parts[ch * pcs_per_ch + k], np.uint8).reshape(-1, STRIPE_BYTES) for k in range(pcs_per_ch)]
    return np.stack(cols, axis=1).reshape(-1)
