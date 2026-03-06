import { Text2VideoPipeline } from 'openvino-genai-node';
import { basename } from 'node:path';

main();

async function main() {
    const modelPath = process.argv[2];
    const prompt = process.argv[3];

    if (!modelPath || !prompt) {
        console.error(`Usage: node ${basename(process.argv[1])} <model_dir> <prompt>`);
        process.exit(1);
    }

    const device = 'CPU'; // GPU can be used as well
    console.log(`Loading model from ${modelPath}...`);
    const pipe = await Text2VideoPipeline(modelPath, device);

    console.log(`Generating video for prompt: "${prompt}"`);
    const result = await pipe.generate(prompt, {
        negative_prompt: 'worst quality, inconsistent motion, blurry, jittery, distorted',
        height: 480,
        width: 704,
        num_frames: 161,
        num_inference_steps: 25,
        num_videos_per_prompt: 1,
        guidance_scale: 3,
        frame_rate: 25,
    });

    const shape = result.video.getShape();
    console.log(`\nGenerated video tensor shape: [${shape.join(', ')}]`);

    const metrics = result.perfMetrics;
    console.log('\nPerformance metrics:');
    console.log(`  Load time: ${metrics.loadTime.toFixed(2)} ms`);
    console.log(`  Generate duration: ${metrics.generateDuration.toFixed(2)} ms`);
    console.log(`  Transformer duration: ${metrics.transformerInferDuration.mean.toFixed(2)} ms`);
    console.log(`  VAE decoder duration: ${metrics.vaeDecoderInferDuration.toFixed(2)} ms`);
}
