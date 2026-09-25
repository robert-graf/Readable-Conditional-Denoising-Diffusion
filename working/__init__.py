import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parent))

from diffusion import Diffusion
