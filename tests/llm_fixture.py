"""Synthetic untrained weights: correctness fixture, never chatbot-quality evidence."""
def tiny():
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from malleable.llm import upstream
    from opentpu.llm.qwen3 import Spec
    torch.manual_seed(7)
    config=Qwen3Config(hidden_size=128,num_hidden_layers=1,num_attention_heads=1,
        num_key_value_heads=1,head_dim=128,intermediate_size=256,vocab_size=128,
        tie_word_embeddings=True,max_position_embeddings=2048)
    model=Qwen3ForCausalLM(config).float().eval()
    return model,{k:v.detach().float().numpy() for k,v in model.state_dict().items()},Spec(128,1,1,1,128,256,128)

def save(path):
    from tokenizers import Tokenizer,models,pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    model,weights,spec=tiny()
    model.save_pretrained(path,safe_serialization=True)
    tokenizer=Tokenizer(models.WordLevel({'<unk>':0,'<eos>':1,'Hello':2,'world':3},unk_token='<unk>'))
    tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token='<unk>',eos_token='<eos>').save_pretrained(path)
    return model,weights,spec
