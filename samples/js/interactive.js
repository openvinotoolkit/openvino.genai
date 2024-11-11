import path from 'node:path';
import readline from 'readline';

import { Pipeline } from 'genai-node';

const MODEL_PATH = process.argv[2];

if (!MODEL_PATH) {
  console.error('Please specify path to model directory\n'
                     + 'Run command must be: `node app.js *path_to_model_dir*`');
  process.exit(1);
}

main();

async function main() {
  // Create interface for reading user input from stdin
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  console.log(`Welcome! Model "${path.parse(MODEL_PATH).name}" loaded. `
              + 'Type something and press enter. Type "finish" to exit.');
  const pipeline = await Pipeline.create('LLMPipeline', MODEL_PATH);
  await pipeline.startChat();
  promptUser();

  // Function to prompt the user for input
  function promptUser() {
    rl.question('> ', handleInput);
  }

  // Function to handle user input
  async function handleInput(input) {
    input = input.trim();

    // Check for exit command
    if (input === 'finish') {
      console.log('Goodbye!');
      await pipeline.finishChat();
      rl.close();
      process.exit(0);
    }

    const result = await pipeline.generate(input, generationCallback);
    console.log('\n');

    // Wait for new input
    promptUser();
  }

  function generationCallback(chunk) {
    process.stdout.write(chunk);
  }
}
