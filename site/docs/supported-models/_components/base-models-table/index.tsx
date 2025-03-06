import Link from '@docusaurus/Link';
import { Children } from 'react';

type BaseModelsTableProps = {
  headers: string[];
  rows: React.JSX.Element[];
};

export function BaseModelsTable({ headers, rows }: BaseModelsTableProps): React.JSX.Element {
  return (
    <table>
      <thead>
        <tr>
          {headers.map((v) => (
            <th key={v}>{v}</th>
          ))}
        </tr>
      </thead>
      <tbody style={{ verticalAlign: 'baseline' }}>{Children.map(rows, (row) => row)}</tbody>
    </table>
  );
}

export const LinksCell = ({ links }: { links: string[] }) => (
  <td>
    <ul>
      {links.map((link) => (
        <li key={link}>
          <Link href={link}>{new URL(link).pathname.slice(1)}</Link>
        </li>
      ))}
    </ul>
  </td>
);

export const StatusCell = ({ value }: { value: boolean }) => (
  <td style={{ textAlign: 'center' }}>{value ? '✅' : '❌'}</td>
);
