import numpy as np
import torch
from test_qwen35_diagnostics import tiny35
from malleable.llm.quality import floating_reference

def test_meta_loaded_reference_restores_rotary_buffers(tiny35,tmp_path):
    weights,spec,expected=tiny35
    info={'family':'qwen35','config':expected.config.to_dict()}
    actual=floating_reference(tmp_path,info,weights)
    assert torch.equal(actual.model.rotary_emb.inv_freq,expected.model.rotary_emb.inv_freq)
    assert actual.model.rotary_emb.inv_freq[0].item()==1.0
    tape=torch.tensor([[3,5,8,13,21]])
    with torch.no_grad():
        wanted=expected(tape,use_cache=False).logits
        got=actual(tape,use_cache=False).logits
    torch.testing.assert_close(got,wanted,rtol=1e-5,atol=1e-6)
    from opentpu.llm.qwen35 import reference_logits
    independent=reference_logits(spec,weights,tape[0].tolist())
    np.testing.assert_allclose(got[0].numpy(),independent,rtol=2e-4,atol=2e-5)

def test_corrected_reference_cache_contract_is_family_specific():
    from malleable.llm.quality import reference_contract
    old='original-local-safetensors-teacher-forced-v1'
    assert reference_contract({'family':'qwen35'})!=old
    assert reference_contract({'family':'qwen3'})==old
    assert reference_contract({'family':'lfm2'})==old
