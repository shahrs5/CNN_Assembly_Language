import re
import struct
import math

def parse_log_consecutive_sequences(log_path, seq_defs):
    """
    seq_defs: dict mapping a name to a list of 3 exact instruction substrings.
    Returns a dict mapping each name to a flat list of all extracted floats.
    """
    with open(log_path, "r") as f:
        lines = [line.strip() for line in f]

    results = {name: [] for name in seq_defs}
    hex_pattern = re.compile(r'([0-9A-Fa-f]{64})')

    for name, seq in seq_defs.items():
        for i in range(len(lines) - 2):
            if all(seq[j] in lines[i + j] for j in range(3)):
                # Found a matching 3-instr block; now grab its next vle32.v
                for k in range(i + 3, len(lines)):
                    if "vle32.v v0, (s0)" in lines[k]:
                        m = hex_pattern.search(lines[k])
                        if m:
                            hex_str = m.group(1)
                            try:
                                floats = [
                                    struct.unpack('>f', bytes.fromhex(hex_str[p:p+8]))[0]
                                    for p in range(0, 64, 8)
                                ]
                            except struct.error:
                                floats = []
                        else:
                            floats = []
                        # Prepend new floats to the front
                        results[name] = floats + results[name]
                        break
    # After collecting, reverse each list before returning
    for name in results:
        results[name].reverse()

    return results


def reshape_and_print(results):
    """
    Prints:
      - input_matrix as a single 28×28 block with header, values rounded up to 2 decimals
      - output_filter as 8 blocks of 24×24 with headers, values rounded up to 2 decimals
      - output_pool as 8 blocks of 12×12 with headers, values rounded up to 2 decimals
      - probabilities as a flat list (no rounding)
    """
    specs = {
        "input_matrix": (28, 28, 1),     # (rows, cols, depth)
        "output_filter": (24, 24, 8),    # 8 filters of 24×24
        "output_pool": (12, 12, 8),      # 8 pools of 12×12
        "probabilities": (10, None, 1),  # 10 values
    }

    for key, (r, c, d) in specs.items():
        data = results.get(key, [])

        total_expected = r * (c or 1) * d
        # Truncate big matrices if needed
        if key != "probabilities" and len(data) > total_expected:
            print(f"[Warning] Truncating {key}: {len(data)} → {total_expected}")
            data = data[-total_expected:]

        # For probabilities, drop last 6 if present
        if key == "probabilities" and len(data) > 6:
            data = data[:-6]

        # Check lengths
        if key != "probabilities":
            assert len(data) == total_expected, \
                f"expected {total_expected} floats for {key}, got {len(data)}"

        print(f"{key.replace('_', ' ').title()} ({len(data)} floats):")

        if key == "probabilities":
            print(data)
        else:
            # iterate depth blocks
            for depth_idx in range(d):
                # Header before each block
                if key == "input_matrix":
                    print(f"Block 28×28:")
                elif key == "output_filter":
                    print(f"24×24 Filter Block {depth_idx + 1}:")
                elif key == "output_pool":
                    print(f"12×12 Pool Block {depth_idx + 1}:")

                block = data[depth_idx * r * c : (depth_idx + 1) * r * c]
                # print each row with ceiling rounding to 2 decimals
                for i in range(r):
                    row = block[i * c:(i + 1) * c]
                    formatted = [f"{math.ceil(v * 100) / 100:.2f}" for v in row]
                    print(formatted)
                # blank line between blocks
                print()

        print("-" * 60 + "\n")


if __name__ == "__main__":
    sequences = {
        "input_matrix": ["c.li     t0, 0x1", "c.li     t0, 0x2", "c.li     t0, 0x3"],
        "output_filter": ["c.li     t0, 0x2", "c.li     t0, 0x3", "c.li     t0, 0x4"],
        "output_pool":   ["c.li     t0, 0x3", "c.li     t0, 0x4", "c.li     t0, 0x5"],
        "probabilities": ["c.li     t0, 0x4", "c.li     t0, 0x5", "c.li     t0, 0x6"],
    }

    results = parse_log_consecutive_sequences("log.txt", sequences)
    reshape_and_print(results)