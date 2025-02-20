import { createRequire } from 'module';
import { platform } from 'node:os';
import { join, dirname, resolve } from 'node:path';

// We need to use delayed import to get an updated Path if required
function getGenAIAddon() {
  const require = createRequire(import.meta.url);
  const ovPath = require.resolve('openvino-node');
  if (platform() == 'win32') {
    // Find the openvino binaries that are required for openvino-genai-node
    const pathToOpenVino = join(dirname(ovPath), '../bin');
    if (!process.env.PATH.includes('openvino-node')) {
      process.env.PATH += ';' + resolve(pathToOpenVino);
    }
  }

  return require('../bin/genai_node_addon.node');
}

export default getGenAIAddon();
