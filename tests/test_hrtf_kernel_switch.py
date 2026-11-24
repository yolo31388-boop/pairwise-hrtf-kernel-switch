import pytest
import media.hrtf_kernel_switch as mod

CASES = [
    ("crossfade_kernel", (1,)),
    ("unified_coord", (1,)),
    ("recompute_itd", (1,)),
    ("load_before_swap", (1,)),
    ("rebuild_filter", (1,)),
    ("hold_old_kernel", (1,)),
    ("select_set", (1,)),
    ("update_source", (1,)),
    ("convolve", (1,)),
    ("swap", (1,)),
    ("validate_spatial", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
