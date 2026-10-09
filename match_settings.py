import json
from runtime_config import ROOT

SETTINGS = json.loads((ROOT / 'rectangle_panel/settings.json').read_text(encoding='utf-8'))
