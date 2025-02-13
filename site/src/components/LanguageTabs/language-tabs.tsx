import styles from "./styles.module.css";
import {CodeExample} from "@site/src/components/CodeExample/code-example";
import {FC} from "react";
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

type LanguageTabsProps = {
    items: {
        title: string;
        content: string;
        language: string;
    }[];
}

export const LanguageTabs: FC<LanguageTabsProps> = ({ items}) => {
    return (
        <Tabs>
            {
                items.map((item) => (
                    <TabItem key={item.title} value={item.language} label={item.title}>
                        <div className={styles.languageContainer}>
                            <CodeExample language={item.language}>
                                {item.content}
                            </CodeExample>
                        </div>
                    </TabItem>
                ))
            }
        </Tabs>
    );
}