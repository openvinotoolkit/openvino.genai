import CodeBlock from '@theme/CodeBlock';

type OptimumCLIProps = {
  model?: string;
  outputDir?: string;
  weightFormat?: 'fp32' | 'fp16' | 'int8' | 'int4';
  task?: string;
  trustRemoteCode?: boolean;
};

export default function OptimumCLI({
  model = '<model_id_or_path>',
  outputDir = '<output_dir>',
  weightFormat,
  task,
  trustRemoteCode,
}: OptimumCLIProps): React.JSX.Element {
  const args = [`--model ${model}`];
  if (weightFormat) {
    args.push(`--weight-format ${weightFormat}`);
  }
  if (task) {
    args.push(`--task ${task}`);
  }
  if (trustRemoteCode) {
    args.push('--trust-remote-code');
  }
  return (
    <CodeBlock language="bash">{`optimum-cli export openvino ${args.join(
      ' '
    )} ${outputDir}`}</CodeBlock>
  );
}
