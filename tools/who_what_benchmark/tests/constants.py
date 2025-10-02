from pathlib import Path
import os
import tempfile


WWB_CACHE_PATH = Path(os.path.join(os.environ.get('OV_CACHE', tempfile.TemporaryDirectory().name), 'wwb_cache'))
SHOULD_CLEANUP = os.environ.get('CLEANUP_CACHE', '').lower() in ('1', 'true', 'yes')
