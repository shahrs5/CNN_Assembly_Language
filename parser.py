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
                        results[name] = floats + results[name]
                        break
    for name in results:
        results[name].reverse()
    return results

def reshape_and_write(results, output_path="output.txt", decimals=3, prob_decimals=2):
    """
    Writes to a txt file:
      - input_matrix as a single 28×28 block with header, values rounded up to `decimals` places
      - output_filter as 8 blocks of 24×24 with headers
      - output_pool as 8 blocks of 12×12 with headers
      - Dense_output as a flat 10-element list rounded to `decimals` places
      - probabilities as a flat list formatted to `prob_decimals` places
    """
    specs = {
        "input_matrix":  (28, 28, 1),
        "output_filter": (24, 24, 8),
        "output_pool":   (12, 12, 8),
        "flattened_pool": (12,12,8),
        "Dense_output":  (10, None, 1),
        "probabilities": (10, None, 1),
    }

    factor      = 10 ** decimals
    prob_factor = 10 ** prob_decimals

    with open(output_path, 'w') as out:
        for key, (r, c, d) in specs.items():
            data = results.get(key, [])
            total_expected = r * (c or 1) * d

            # Truncate if too-long
            if key != "probabilities" and key != "Dense_output" and len(data) > total_expected:
                out.write(f"[Warning] Truncating {key}: {len(data)} → {total_expected}\n")
                data = data[-total_expected:]

            # Special-case probabilities drop-tail
            if key == "probabilities" and len(data) > 6:
                data = data[:-6]

            # Special-case probabilities drop-tail
            if key == "Dense_output" and len(data) > 6:
                data = data[:-6]

            # sanity check for everything except probs
            if key not in ("probabilities", "Dense_output"):
                assert len(data) == total_expected, \
                    f"expected {total_expected} floats for {key}, got {len(data)}"

            out.write(f"{key.replace('_', ' ').title()} ({len(data)} floats):\n")

            # === PROBABILITIES ===
            if key == "probabilities":
                formatted = [
                    f"{math.ceil(v * prob_factor) / prob_factor:.{prob_decimals}f}"
                    for v in data
                ]
                out.write(str(formatted) + "\n\n")

            # === DENSE OUTPUT ===
            elif key == "Dense_output":
                # 10 elements, round up to `decimals` places
                formatted = [
                    f"{math.ceil(v * factor) / factor:.{decimals}f}"
                    for v in data
                ]
                out.write(str(formatted) + "\n\n")

            # === ALL OTHER 2-D BLOCKS ===
            else:
                for depth_idx in range(d):
                    if key == "input_matrix":
                        out.write(f"Block 28×28:\n")
                    elif key == "output_filter":
                        out.write(f"24×24 Filter Block {depth_idx + 1}:\n")
                    elif key == "output_pool":
                        out.write(f"12×12 Pool Block {depth_idx + 1}:\n")

                    block = data[depth_idx * r * c : (depth_idx + 1) * r * c]
                    for i in range(r):
                        row = block[i * c:(i + 1) * c]
                        row_fmt = [
                            f"{math.ceil(v * factor) / factor:.{decimals}f}"
                            for v in row
                        ]
                        out.write(str(row_fmt) + "\n")
                    out.write("\n")

            out.write("-" * 60 + "\n\n")



if __name__ == "__main__":
    sequences = {
        "input_matrix": ["c.li     t0, 0x1", "c.li     t0, 0x2", "c.li     t0, 0x3"],
        "output_filter": ["c.li     t0, 0x2", "c.li     t0, 0x3", "c.li     t0, 0x4"],
        "output_pool":   ["c.li     t0, 0x3", "c.li     t0, 0x4", "c.li     t0, 0x5"],
        "flattened_pool":   ["c.li     t0, 0x4", "c.li     t0, 0x5", "c.li     t0, 0x6"],
        "Dense_output": ["c.li     t0, 0x5", "c.li     t0, 0x6", "c.li     t0, 0x7"],
        "probabilities": ["c.li     t0, 0x6", "c.li     t0, 0x7", "c.li     t0, 0x8"],
    }

    results = parse_log_consecutive_sequences("log.txt", sequences)
    reshape_and_write(results, output_path="output.txt", decimals=3, prob_decimals=2)
    print("Results written to output.txt")