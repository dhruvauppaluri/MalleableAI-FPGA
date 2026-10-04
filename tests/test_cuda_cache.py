"""Opt-in, checkpoint-free cache tests on an actual CUDA device, never CPU fallback."""
import os
import pytest

pytestmark=pytest.mark.skipif(os.environ.get('MALLEABLE_CUDA_CACHE_TESTS')!='1',
    reason='actual CUDA cache tests require explicit MALLEABLE_CUDA_CACHE_TESTS=1')


@pytest.fixture
def verifier():
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from malleable.llm.hybrid import CudaVerifier
    if not torch.cuda.is_available(): pytest.fail('actual CUDA requested but unavailable')
    torch.manual_seed(0); torch.backends.cuda.matmul.allow_tf32=False
    cfg=Qwen3Config(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=16,max_position_embeddings=128)
    cfg._attn_implementation='eager'
    v=object.__new__(CudaVerifier); v.torch=torch
    v.model=Qwen3ForCausalLM(cfg).half().cuda().eval()
    v.cache=None; v.next=None; v.position=0; v.pending=None
    return v


@pytest.mark.parametrize('depth',[1,2,4,8])
@pytest.mark.parametrize('boundary',[False,True])
def test_cuda_crop_and_correction_at_every_rejection_position(verifier,depth,boundary):
    prompt=[1]*(128-depth if boundary else 3)
    verifier.reset(prompt); expected=[]
    for _ in range(depth):
        token=verifier.next_token(); expected.append(token); verifier.advance(token)
    for rejected_at in range(depth):
        verifier.reset(prompt)
        block=expected.copy(); block[rejected_at]=(block[rejected_at]+1)%128
        predicted=verifier.verify(block)
        assert predicted[:rejected_at+1]==expected[:rejected_at+1]
        verifier.commit(rejected_at,expected[rejected_at])
        assert verifier.position==len(prompt)+rejected_at+1
        assert verifier.cache.get_seq_length()==verifier.position
        next_token=verifier.next_token()
        verifier.reset(prompt+expected[:rejected_at+1])
        assert verifier.next_token()==next_token


@pytest.mark.parametrize('depth',[1,2,4,8])
def test_cuda_greedy_agreement_and_eos(verifier,depth):
    from malleable.llm.hybrid import greedy
    from test_hybrid import FakeDraft
    verifier.reset([1]); expected=[]
    for _ in range(8):
        token=verifier.next_token(); expected.append(token); verifier.advance(token)
    result=greedy(FakeDraft(expected),verifier,[1],max_new=8,depth=depth)
    assert result['tokens']==expected
    eos=expected[0]
    result=greedy(FakeDraft([(eos+1)%128]*8),verifier,[1],max_new=8,depth=depth,eos={eos})
    assert result['tokens']==[eos]
