import { Pipeline } from 'genai-node';

const MODEL_PATH = process.argv[2];

if (!MODEL_PATH) {
  console.error('Please specify path to model directory\n'
                    + 'Run command must be: `node app.js *path_to_model_dir*`');
  process.exit(1);
}

const generationCallback = (chunk) => {
  process.stdout.write(chunk);
};

const prompt = 'Who are you?';
console.log(`User Prompt: "${prompt}"\n`);

const pipeline = await Pipeline.create('LLMPipeline', MODEL_PATH);

await pipeline.startChat();
const result = await pipeline.generate(
  prompt,
  generationCallback,
  { temperature: 0 },
);
await pipeline.finishChat();

console.log(`\n\nGeneration result:\n"${result}"`)
