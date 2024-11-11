import { testModels, downloadTestModel } from './utils.js';

if (require.main === module) {
  main();
}

async function main() {
  await downloadTestModel(testModels.testModelFP32);
}
