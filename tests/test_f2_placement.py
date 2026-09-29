"""F2 address map: pure-software checks (no simulator)."""
import numpy as np
import pytest

from malleable.f2 import placement as P


def test_window_addresses():
    assert P.pcis_address(0, 0) == 1 << 36
    assert P.pcis_address(1, 0x1000) == (1 << 36) + 0x8000_0000 + 0x1000
    assert P.in_window(P.pcis_address(1, P.CH_BYTES - 1))
    assert not P.in_window(P.HBM_BASE - 1)
    assert not P.in_window(P.HBM_BASE + P.WINDOW_BYTES)
    with pytest.raises(ValueError):
        P.pcis_address(2, 0)
    with pytest.raises(ValueError):
        P.pcis_address(0, P.CH_BYTES)


@pytest.mark.parametrize("pcs", [2, 4, 8])
def test_placement_is_a_bijection_and_stripes_rotate(pcs):
    n = pcs * 512 * 6
    seen = set()
    for ch in (0, 1):
        for off in range(0, n, 64):
            pc, local = P.place(ch, off, pcs)
            assert ch * pcs <= pc < (ch + 1) * pcs
            assert (pc, local) not in seen
            seen.add((pc, local))
    # consecutive 512-byte stripes rotate over the channel's PCs
    assert [P.place(0, k * 512, pcs)[0] for k in range(pcs + 1)] == list(range(pcs)) + [0]
    # inside a stripe the PC does not change and local addresses are contiguous
    pc0, l0 = P.place(1, 3 * 512, pcs)
    assert P.place(1, 3 * 512 + 448, pcs) == (pc0, l0 + 448)


def test_rtl_reference_points():
    # values checked by hand against f2/sim/tb_f2_adapter.sv (PCS = 2)
    assert P.place(0, 2048, 2) == (0, 1024)      # beat 32 of channel 0
    assert P.place(0, 24 * 64, 2) == (1, 512)    # beat 24: stripe 3 -> PC 1
    assert P.place(1, 24 * 64, 2) == (3, 512)


def test_out_of_range_matches_router_decerr():
    P.place(0, (2 << 29) - 1, 2, 29)
    with pytest.raises(ValueError):
        P.place(0, 2 << 29, 2, 29)              # the router answers DECERR here
    assert P.channel_capacity(2, 29) == 1 << 30
    assert P.channel_capacity(16, 29) == P.CH_BYTES


def test_vectorized_matches_scalar():
    offs = np.arange(0, 512 * 9, 64)
    pcs, local = P.place_many(1, offs, 4)
    for o, a, b in zip(offs, pcs, local):
        assert P.place(1, int(o), 4) == (int(a), int(b))


@pytest.mark.parametrize("pcs", [2, 4])
def test_split_join_roundtrip(pcs):
    data = np.random.default_rng(3).integers(0, 256, 512 * pcs * 5, dtype=np.uint8)
    parts = P.split_channel(data, 1, pcs)
    assert sorted(parts) == list(range(pcs, 2 * pcs))
    assert all(len(v) == len(data) // pcs for v in parts.values())
    assert np.array_equal(P.join_channel(parts, 1, pcs), data)
    # each stripe is one contiguous run inside its PC
    assert np.array_equal(parts[pcs][:512], data[:512])
    assert np.array_equal(parts[pcs + 1][:512], data[512:1024])
    assert np.array_equal(parts[pcs][512:1024], data[512 * pcs:512 * pcs + 512])
    with pytest.raises(ValueError):
        P.split_channel(data[:-1], 0, pcs)
