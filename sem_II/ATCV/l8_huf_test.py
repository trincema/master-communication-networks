from collections import Counter
import heapq

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def rle_encode(data):
    encoding = []
    prev_char = data[0]
    count = 1

    for char in data[1:]:
        if char == prev_char:
            count += 1
        else:
            encoding.append((prev_char, count))
            prev_char = char
            count = 1

    encoding.append((prev_char, count))
    return encoding

def build_huffman_tree(freq_map):
    heap = [Node(char, freq) for char, freq in freq_map.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = Node(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heap[0]

def build_huffman_codes(node, prefix="", codebook={}):
    if node is not None:
        if node.char is not None:
            codebook[node.char] = prefix
        build_huffman_codes(node.left, prefix + "0", codebook)
        build_huffman_codes(node.right, prefix + "1", codebook)
    return codebook

def huffman_encode(data, codebook):
    encoded_data = ""
    for item in data:
        encoded_data += codebook[item]
    return encoded_data

# Example usage:
image_block = [1, 1, 1, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 3, 0, 0, 0]
rle_encoded_block = rle_encode(image_block)
print("RLE Encoded Block:", rle_encoded_block)

# Flatten the RLE encoded block for frequency analysis
rle_flattened = [item for sublist in rle_encoded_block for item in sublist]

# Build frequency map and Huffman tree
freq_map = Counter(rle_flattened)
huffman_tree = build_huffman_tree(freq_map)
huffman_codes = build_huffman_codes(huffman_tree)
print("Huffman Codes:", huffman_codes)

# Encode the RLE data using Huffman codes
encoded_rle_data = huffman_encode(rle_flattened, huffman_codes)
print("Huffman Encoded RLE Data:", encoded_rle_data)
