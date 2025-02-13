import {Prism as SyntaxHighlighter} from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import {FC} from "react";

import './code-example.css';

type CodeExampleProps = {
    language: string;
    children: string;
}

export const CodeExample:FC<CodeExampleProps> = ({language, children}) => {
    return (
        <SyntaxHighlighter language={language} style={oneLight} wrapLines useInlineStyles={false}>
            {children}
        </SyntaxHighlighter>
    );
}