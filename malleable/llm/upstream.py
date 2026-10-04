"""Source-checkout integration; upstream code remains separately attributed."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2] / 'third_party' / 'opentpu'
REVISION = '15754e971b55591b91048c4c636023fe59b343e7'
if not (ROOT / 'UPSTREAM.json').exists():
    raise RuntimeError('OpenTPU assets require the complete source checkout')
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
