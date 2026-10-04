import unittest
from malleable.llm.hybrid import greedy

class FakeDraft:
    def __init__(self,sequence): self.sequence=sequence; self.position=0
    def reset(self,tokens): self.position=len(tokens)-1
    def next_token(self): return self.sequence[self.position%len(self.sequence)]
    def advance(self,token): self.position+=1

class FakeVerifier(FakeDraft):
    def verify(self,block):
        return [self.sequence[(self.position+i)%len(self.sequence)] for i in range(len(block))]
    def commit(self,n,correction): self.position+=n+int(correction is not None)

class HybridTests(unittest.TestCase):
    def test_accept_reject_positions_and_depths(self):
        target=[2,3,4,5,6,7,8,9]
        for proposal in (target,[99]*8,[2,99,4,5,6,7,8,9],[2,3,99,5,6,7,8,9]):
            for depth in (1,2,4,8):
                v=FakeVerifier(target)
                result=greedy(FakeDraft(proposal),v,[1],max_new=8,depth=depth)
                self.assertEqual(result['tokens'],target); self.assertEqual(v.position,8)
    def test_eos_cancel_context(self):
        result=greedy(FakeDraft([2,3,4]),FakeVerifier([2,3,4]),[1],max_new=8,depth=8,eos=3)
        self.assertEqual(result['tokens'],[2,3])
        with self.assertRaises(InterruptedError): greedy(FakeDraft([2]),FakeVerifier([2]),[1],cancel=lambda:True)
        with self.assertRaises(ValueError): greedy(FakeDraft([2]),FakeVerifier([2]),[1],depth=3)
    def test_cpu_not_cuda(self):
        import torch
        if not torch.cuda.is_available():
            from malleable.llm.hybrid import CudaVerifier
            with self.assertRaisesRegex(ValueError,'CUDA required'): CudaVerifier('not-a-model')

if __name__=='__main__': unittest.main()
