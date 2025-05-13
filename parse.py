import re
import struct

def parse_log_consecutive_sequences(log_path, seq_defs):
    """
    seq_defs: dict mapping a name (e.g. "input_matrix")
              to a list of 3 exact instruction substrings.
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
                                # Big-endian unpacking, insert floats at front
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
    specs = {
        "input_matrix": (28 * 28, 28),
        "output_filter": (8 * 24 * 24, 24),
        "output_pool": (8 * 12 * 12, 12),
        "probabilities": (10, None),
    }

    for key, (expected, row_len) in specs.items():
        data = results.get(key, [])

        # Truncate only the big matrices if they overflow
        if key != "probabilities" and len(data) > expected:
            print(f"[Warning] Truncating {key}: {len(data)} → {expected}")
            data = data[-expected:]
        
        # For probabilities, truncate the *last six* values
        if key == "probabilities":
            if len(data) > 6:
                data = data[:-6]  # Remove last six values

        # For input/filter/pool we assert exact length; for probabilities we skip that
        if key != "probabilities":
            assert len(data) == expected, f"expected {expected} floats for {key}, got {len(data)}"

        print(f"{key.replace('_', ' ').title()} ({len(data)} floats):")

        if row_len:
            groups = len(data) // row_len
            for g in range(groups):
                print(data[g * row_len : (g + 1) * row_len])
        else:
            print(data)
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
