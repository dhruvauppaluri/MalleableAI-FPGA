"""Independent, arbitrary-precision integer contract for the SSM prototype.

Prototype activations/weights are signed Q2.14 stored in 32-bit memory words.
This deliberately precedes production INT8/INT4 group quantization.
"""
import math

FRACTION = 14
ONE = 1 << FRACTION


def round_shift(value, shift=FRACTION):
    if not 0 <= shift <= 62:
        raise ValueError('shift must be in 0..62')
    if not shift:
        return value
    return (1 if value >= 0 else -1) * ((abs(value) + (1 << (shift - 1))) >> shift)


def sat(value):
    return max(-32768, min(32767, value))


def quantize(value):
    return sat(int(math.copysign(math.floor(abs(value) * ONE + .5), value)))


def execute(memory, program):
    """Execute immutable-input instructions against a copied memory image.

    Each instruction: [opcode,dst,a,b,length,aux,shift,bias_address].
    DOT uses bias_address (pre-scaled Q14); aux is a table address for LUT,
    or Q14 epsilon for RMS. An END instruction has opcode 255.
    Saturation counts are observable and disqualify quality promotion by default.
    """
    memory = list(memory)
    saturated = 0
    for op, dst, a, b, length, aux, shift, bias in program:
        if op == 255:
            return memory, saturated
        if op not in range(6) or not 1 <= length <= 4096 or not 0 <= shift <= 62:
            raise ValueError('invalid instruction')
        out_count = 1 if op == 0 else length
        for addr, count in [(dst, out_count), (a, length)]:
            if addr < 0 or addr + count > len(memory):
                raise ValueError('instruction out of bounds')
        if op in (0, 1, 2) and (b < 0 or b + length > len(memory)):
            raise ValueError('second operand out of bounds')
        if op == 0 and not 0 <= bias < len(memory):
            raise ValueError('bias out of bounds')
        if op == 3 and not 0 <= aux <= len(memory) - 256:
            raise ValueError('lookup out of bounds')
        # No overlapping destinations: a hardware instruction consumes source
        # words over several cycles. The compiler allocates distinct buffers.
        sources = [(a, length)] + ([(b, length)] if op in (0, 1, 2) else [])
        if any(dst < start + count and start < dst + out_count for start, count in sources):
            raise ValueError('overlapping instruction operands')
        av = memory[a:a + length]
        bv = memory[b:b + length]
        if op == 0:
            values = [round_shift(sum(x*y for x,y in zip(av,bv)),shift) + memory[bias]]
        elif op == 1:
            values = [x + y for x,y in zip(av,bv)]
        elif op == 2:
            values = [round_shift(x*y,shift) for x,y in zip(av,bv)]
        elif op == 3:
            values = [memory[aux + max(0,min(255,(x >> shift) + 128))] for x in av]
        elif op == 4:
            root = math.isqrt(sum(x*x for x in av)//length + aux * ONE)
            root = max(1,root)
            values = [round_ratio(x * ONE,root) for x in av]
        else:
            values = av
        for i,value in enumerate(values):
            saturated += int(value < -32768 or value > 32767)
            memory[dst+i] = sat(value)
    raise ValueError('program has no END')


def round_ratio(value, divisor):
    return (1 if value >= 0 else -1) * ((abs(value) + divisor//2)//divisor)
