from pathlib import Path
import os
import tempfile


WWB_CACHE_PATH = Path(os.path.join(os.environ.get('OV_CACHE', tempfile.TemporaryDirectory()), 'wwb_cache'))
SHOULD_CLEANUP = bool(os.environ.get('CLEANUP_CACHE', None))
