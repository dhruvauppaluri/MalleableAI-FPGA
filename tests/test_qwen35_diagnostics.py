"""Family controls exercise the existing reference with synthetic, untrained weights."""
import numpy as np
import pytest
from malleable.llm.precision import GROUPS,QWEN35_SITES,controlled_reference,OperatorStats
from malleable.llm.diagnostics import _captured_qwen35

@pytest.fixture(scope='module')
def tiny35():
    import torch
    from transformers import Qwen3_5TextConfig,Qwen3_5ForCausalLM
    from malleable.llm import upstream
    from opentpu.llm.qwen35 import Spec
    torch.manual_seed(33)
    config=Qwen3_5TextConfig(hidden_size=128,num_hidden_layers=2,num_attention_heads=2,
        num_key_value_heads=1,head_dim=256,intermediate_size=256,vocab_size=128,
        layer_types=['linear_attention','full_attention'],linear_num_key_heads=2,
        linear_num_value_heads=2,linear_key_head_dim=128,linear_value_head_dim=128,
        linear_conv_kernel_dim=4,tie_word_embeddings=True,max_position_embeddings=2048,
        rms_norm_eps=1e-6,rope_parameters={'rope_type':'default','rope_theta':1e7,'partial_rotary_factor':.25})
    model=Qwen3_5ForCausalLM(config).float().eval()
    with torch.no_grad():
        for name,parameter in model.named_parameters():
            if 'norm' in name:parameter.copy_((1. if name.endswith('linear_attn.norm.weight') else 0.)+.1*torch.randn_like(parameter))
    weights={k:v.detach().numpy() for k,v in model.state_dict().items()}
    return weights,Spec(128,('linear','attn'),2,1,256,64,2,128,128,256,128)

def policy(bypass=()):return {'schema_version':1,'quantized_groups':[g for g in GROUPS if g not in bypass],'group_size':128}

def test_qwen35_baseline_sites_and_capture_preserve_output(tiny35):
    from opentpu.llm.qwen35 import emulated_logits,reference_logits
    weights,spec=tiny35;tape=[3,5,8,13,21]
    original=emulated_logits(spec,weights,tape)
    stats=OperatorStats();forward,_=controlled_reference(policy(),stats,'qwen35')
    assert np.array_equal(forward(spec,weights,tape),original)
    assert np.array_equal(forward(spec,weights,tape),original) # no recurrent state leak
    assert {row['operator'] for row in stats.export() if row['group'] not in ('transformer_weights','head_weights')}=={site for _,site,_ in QWEN35_SITES}
    captures={};actual=_captured_qwen35(emulated_logits,captures)(spec,weights,tape)
    assert np.array_equal(actual,original)
    stages={key[0] for key in captures}
    assert {'convolution','deltanet_state','attention_output','input_normalization','mlp_normalization',
        'mlp_residual','beta_gate','decay_gate','output_gate','head_normalization'}<=stages
    captures={};actual=_captured_qwen35(reference_logits,captures)(spec,weights,tape)
    assert np.array_equal(actual,reference_logits(spec,weights,tape))

@pytest.mark.parametrize('bypass',[(),GROUPS,('transformer_weights',),('projection_inputs',),('key_cache',),
    ('value_cache',),('attention',),('head_weights',),('head_inputs',),('transformer_weights','projection_inputs')])
def test_ten_controls_are_finite_and_account_for_every_group(tiny35,bypass):
    weights,spec=tiny35;stats=OperatorStats();forward,_=controlled_reference(policy(bypass),stats,'qwen35')
    logits=forward(spec,weights,[3,5,8])
    assert logits.shape==(3,spec.vocab) and np.isfinite(logits).all()
    rows=stats.export()
    assert {r['group'] for r in rows}==set(GROUPS)
    assert all(r['quantized']==(r['group'] not in bypass) for r in rows)
    assert all(r['error_squared']==0 for r in rows if r['group'] in bypass)
