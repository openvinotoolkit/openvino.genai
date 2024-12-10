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

  // Collect wheel paths
  const wheelPaths = [];
  for (const pkg of packages) {
    const packageName = pkg.split('[')[0];
    if (localWheels[packageName]) {
      const wheelPath = localWheels[packageName];
      wheelPaths.push(`"${wheelPath}${pkg.slice(packageName.length)}"`);
    } else {
      core.setFailed(`Package ${pkg} not found locally.`);
      return;
    }
  }

  // Install all wheels in one command
  if (wheelPaths.length > 0) {
    console.log(`Installing local wheels: ${wheelPaths.join(' ')}`);
    const { stdout, stderr } = await execAsync(
      `pip install ${wheelPaths.join(' ')}`,
      {
        stdio: 'inherit'
      }
    );
    console.log('stdout:', stdout);
    console.error('stderr:', stderr);
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
