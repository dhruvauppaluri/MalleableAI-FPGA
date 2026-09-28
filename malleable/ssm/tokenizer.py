class Codec:
    BOS,EOS,PAD = 256,257,258

    def __init__(self, metadata=None):
        metadata = metadata or {'kind':'byte-v1'}
        self.kind = metadata['kind']
        self.tokenizer = None
        if self.kind == 'tokenizers-json':
            import json
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_str(json.dumps(metadata['json']))
            self.BOS=metadata.get('bos_id',0)
            self.EOS=metadata.get('eos_id',0)
            self.PAD=metadata.get('pad_id',self.EOS)
        elif self.kind != 'byte-v1':
            raise ValueError('unsupported tokenizer')

    def encode(self,text):
        return self.tokenizer.encode(text,add_special_tokens=False).ids if self.tokenizer else list(text.encode('utf-8'))

    def decode(self,tokens):
        if self.tokenizer:
            return self.tokenizer.decode(tokens,skip_special_tokens=True)
        return bytes(t for t in tokens if 0 <= t < 256).decode('utf-8',errors='replace')
