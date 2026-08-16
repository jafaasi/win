from typing import Dict, Any

def map_digit(digit: int) -> Dict[str, int]:
    """
    Canonical deterministic mapping from integer digit 0..9 to size, color, and parity:
    - size: 1 if digit >= 5 (Big) else 0 (Small)
    - color: 0 = Green (1, 3, 7, 9), 1 = Red (2, 4, 6, 8), 2 = Violet (0, 5)
    - parity: 0 = Even, 1 = Odd
    """
    if not isinstance(digit, int) or digit < 0 or digit > 9:
        raise ValueError(f"digit must be an integer between 0 and 9, got {digit}")
        
    size = 1 if digit >= 5 else 0
    if digit in (1, 3, 7, 9):
        color = 0  # Green
    elif digit in (2, 4, 6, 8):
        color = 1  # Red
    else:
        color = 2  # Violet
        
    parity = digit % 2
    return {
        "digit": digit,
        "size": size,
        "color": color,
        "parity": parity
    }
