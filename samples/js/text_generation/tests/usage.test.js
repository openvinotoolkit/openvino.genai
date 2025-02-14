import { env } from 'process';
import { spawn } from 'child_process';

const MODEL_PATH = env.MODEL_PATH;
const prompt = 'Tell me exactly, no changes, print as is: "Hello world"';

if (!MODEL_PATH)
  throw new Error(
    'Please environment variable MODEL_PATH to the path of the model directory'
  );

const runTest = async () => {
  return new Promise((resolve, reject) => {
    const script = spawn('node', ['chat_sample.js', MODEL_PATH]);
    let output = '';

    // Collect output from stdout
    script.stdout.on('data', (data) => {
      output += data.toString();
    });

    // Capture errors
    script.stderr.on('data', (data) => {
      reject(data.toString());
    });

    // Send input after detecting the question prompt
    script.stdout.once('data', (data) => {
      if (data.toString().startsWith('question:')) {
        script.stdin.write(`${prompt}\n`); // Provide input
        script.stdin.end(); // Close stdin to signal EOF
      }
    });

    // Check results when the process exits
    script.on('close', (code) => {
      if (code !== 0) {
        return reject(`Process exited with code ${code}`);
      }

      // Log the output
      console.log(`Result output: ${output}`);

      // Validate the output
      if (typeof output == 'string' && output.length > 0) {
        resolve('Test passed!');
      } else {
        reject('Test failed: Output did not match expected result.');
      }
    });
  });
};

runTest()
  .then((message) => {
    console.log(message);
    process.exit(0);
  })
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
