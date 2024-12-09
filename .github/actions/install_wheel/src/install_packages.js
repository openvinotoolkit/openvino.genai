const core = require('@actions/core');
const glob = require('glob');
const path = require('path');
const { exec } = require('child_process');
const util = require('util');

const execAsync = util.promisify(exec);

async function installPackages(packages, localWheelDir) {
  // Resolve local wheels
  const localWheels = {};
  if (localWheelDir) {
    const wheels = glob.sync(path.join(localWheelDir, '*.whl'));
    for (const whl of wheels) {
      const packageName = path.basename(whl).split('-')[0];
      localWheels[packageName] = whl;
    }
  }

  // Install packages
  for (const pkg of packages) {
    const packageName = pkg.split('[')[0];
    if (localWheels[packageName]) {
      const wheelPath = localWheels[packageName];
      console.log(`Installing local wheel: ${wheelPath}`);
      await execAsync(
        `pip install "${wheelPath}${pkg.slice(packageName.length)}"`,
        {
          stdio: 'inherit'
        }
      );
    } else {
      console.log(`Installing from PyPI: ${pkg}`);
      await execAsync(`pip install "${pkg}"`, { stdio: 'inherit' });
    }
  }
}

async function run() {
  try {
    const packagesInput = core.getInput('packages');
    const localWheelDir = core.getInput('local_wheel_dir') || null;
    const packages = packagesInput.split(';');
    await installPackages(packages, localWheelDir);
  } catch (error) {
    core.setFailed(error.message);
  }
}

run();
