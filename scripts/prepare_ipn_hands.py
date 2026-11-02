"""CLI entry point for the official IPN HandS -> project IPN-13 adapter."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "final" / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src_neurosymbolic.pose.ipn_hands_adapter import main  # noqa: E402


if __name__ == "__main__":
    main()
