import readline from 'readline';
import { LLMPipeline } from 'openvino-genai-node';
import { basename } from 'node:path';

main();

function streamer(subword) {
  process.stdout.write(subword);
}

async function main() {
  const MODEL_PATH = process.argv[2];

  if (process.argv.length > 3) {
    console.error(`Run command must be: 'node ${basename(process.argv[1])} *path_to_model_dir*'`);
    process.exit(1);
  }
  if (!MODEL_PATH) {
    console.error('Please specify path to model directory\n'
                  + `Run command must be: 'node ${basename(process.argv[1])} *path_to_model_dir*'`);
    process.exit(1);
  }

  const device = 'CPU'; // GPU can be used as well

  // Create interface for reading user input from stdin
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const pipe = await LLMPipeline(MODEL_PATH, device);
  const config = { 'max_new_tokens': 100 };

  await pipe.startChat();
  promptUser();

  // Function to prompt the user for input
  function promptUser() {
    rl.question('question:\n', handleInput);
  }

  // Function to handle user input
  async function handleInput(input) {
    input = input.trim();

    // Check for exit command
    if (!input) {
      await pipe.finishChat();
      rl.close();
      process.exit(0);
    }

    await pipe.generate(input, config, streamer);
    console.log('\n----------');

    if (!rl.closed) promptUser();
  }
}
