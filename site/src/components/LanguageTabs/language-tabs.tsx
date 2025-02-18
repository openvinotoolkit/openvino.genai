import {FC} from "react";
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import CodeBlock from '@theme/CodeBlock';

type LanguageTabsProps = {
    items: {
        title: string;
        content: string;
        language: string;
    }[];
}

export const LanguageTabs: FC<LanguageTabsProps> = ({ items }) => {
    return (
        <Tabs>
            {
                items.map((item) => (
                    <TabItem key={item.title} value={item.language} label={item.title}>
                        <CodeBlock language={item.language} className='language-tabs__code-block'>
                            {item.content}
                        </CodeBlock>
                    </TabItem>
                ))
            }
        </Tabs>
    );
}