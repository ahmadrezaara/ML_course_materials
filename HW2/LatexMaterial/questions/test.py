def generate_bitstreams(k):
    bitstreams = []
    for i in range(2 ** k):
        bitstream = bin(i)[2:].zfill(k)
        bitstreams.append(bitstream)
    return bitstreams

def process_bitstream(bitstream):
    sum_ = 0
    for bit in bitstream:
        if bit == '1':
            sum_ += 1
        else:
            sum_ -= 1
        if sum_ == 0:
            return 1
    return 0

def main():
    list1 = []
    k = int(input("Enter the length of bitstreams (k): "))
    bitstreams = generate_bitstreams(k)
    final_sum = 0
    sum2 = 0
    for bitstream in bitstreams:
        final_sum += process_bitstream(bitstream)
        if(process_bitstream(bitstream) == 1):
            list1.append(bitstream)
    print("Final sum:", final_sum/(2**k))
    for i in range(k) :
        if i%2 == 1:
            sum2 = sum2 + 0.5**i
    print("Sum2:",sum2)
if __name__ == "__main__":
    main()