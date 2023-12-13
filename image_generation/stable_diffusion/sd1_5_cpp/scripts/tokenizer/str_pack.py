import numpy as np

# Convert any list of string to U8/1D numpy array compatible with converted OV model input
def pack_strings(strings):
    to_bytes = lambda x: x.to_bytes(4, 'little')
    batch_size = len(strings)
    if batch_size == 0:
        return to_bytes(0)
    offsets = to_bytes(0)
    symbols = bytes()
    for s in strings:
        symbols += bytes(s, 'utf-8')
        offsets += to_bytes(len(symbols))
    return np.frombuffer(bytearray(to_bytes(batch_size) + offsets + symbols), np.uint8)

# Convert an array of uint8 elements to a list of strings; reverse to pack_strings
# TODO: handle possible sighed values in batch size and offsets
def unpack_strings(u8_tensor):
    from_bytes = lambda offset, size: int.from_bytes(u8_tensor[offset:offset+size], 'little')
    batch_size = from_bytes(0, 4)
    strings = []
    for i in range(batch_size):
        begin = from_bytes(4 + i*4, 4)
        end = from_bytes(4 + (i+1)*4, 4)
        length = end - begin
        begin += 4*(batch_size + 2)
        strings.append(bytes(u8_tensor[begin:begin+length]).decode('utf-8'))
    return strings
