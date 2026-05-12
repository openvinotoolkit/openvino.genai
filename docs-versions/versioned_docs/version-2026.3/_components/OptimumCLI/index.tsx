import CodeBlock from '@theme/CodeBlock';

type OptimumCLIProps = {
  model?: string;
  outputDir?: string;
  weightFormat?: 'fp32' | 'fp16' | 'int8' | 'int4';
  task?: string;
  trustRemoteCode?: boolean;
  modelKwargs?: Record<string, string>;
  quantMode?: 'int8';
  dataset?: string;
  numSamples?: number;
};

export default function OptimumCLI({
  model = '<model_id_or_path>',
  outputDir = '<output_dir>',
  weightFormat,
  task,
  trustRemoteCode,
  modelKwargs,
  quantMode,
  dataset,
  numSamples,
}: OptimumCLIProps): React.JSX.Element {
  const args = [`--model ${model}`];
  if (weightFormat) {
    args.push(`--weight-format ${weightFormat}`);
  }
  if (task) {
    args.push(`--task ${task}`);
  }
  if (quantMode) {
    args.push(`--quant-mode ${quantMode}`);
  }
  if (dataset) {
    args.push(`--dataset ${dataset}`);
  }
  if (numSamples) {
    args.push(`--num-samples ${numSamples}`);
  }
  if (trustRemoteCode) {
    args.push('--trust-remote-code');
  }
  if (modelKwargs) {
    const kwargsString = JSON.stringify(modelKwargs);
    args.push(`--model-kwargs '${kwargsString}'`);
  }
  return (
    <CodeBlock language="bash">{`optimum-cli export openvino ${args.join(
      ' '
    )} ${outputDir}`}</CodeBlock>
  );
}
