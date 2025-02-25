import { dowloadModel } from './utils.js';
import { models } from './models.js';

for (const model of models) {
  await dowloadModel(model);
}
