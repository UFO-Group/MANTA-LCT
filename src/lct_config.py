# lct_config.py

def parse_value(token: str):
    """
    Convert a string token into int, float, or an unquoted string.

    Rules:
        - If token is wrapped in quotes ("..."/'...'), remove the quotes.
        - Try to convert to int.
        - Try to convert to float.
        - Otherwise return the original string.
    """
    token = token.strip()

    # Remove matching quotes
    if (token.startswith('"') and token.endswith('"')) or \
       (token.startswith("'") and token.endswith("'")):
        return token[1:-1]

    # Try integer
    try:
        return int(token)
    except ValueError:
        pass

    # Try float
    try:
        return float(token)
    except ValueError:
        pass

    # Fallback: treat as plain string
    return token


def load_lct_input(filename: str):
    """
    Load a simple key-value input file (lct.in or paths.cfg).

    Parsing rules:
        - Empty lines or lines starting with '#' are ignored.
        - A line with only a key (no values) → treated as a boolean flag (True).
        - A key followed by one value → parsed by parse_value().
        - A key followed by multiple values → stored as a list.

    Returns:
        dict : A dictionary mapping keys (lowercased) to parsed values.
    """
    config = {}

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty or commented lines
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            key = parts[0].lower()
            vals = parts[1:]

            if not vals:
                config[key] = True  # Boolean flag
            elif len(vals) == 1:
                config[key] = parse_value(vals[0])
            else:
                config[key] = [parse_value(v) for v in vals]

    return config
