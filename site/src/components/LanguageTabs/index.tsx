import TabItem, { Props as TabItemProps } from '@theme/TabItem';
import Tabs from '@theme/Tabs';
import { Children, ReactElement } from 'react';

type LanguageTabsProps = {
  children: ReactElement<LanguageTabItemProps, typeof TabItemPython | typeof TabItemCpp>[];
};

export function LanguageTabs({ children, ...props }: LanguageTabsProps) {
  const tabsChildren = Children.map(children, (child) => {
    if (child.type === TabItemPython) {
      return TabItemPython(child.props as TabItemProps);
    } else if (child.type === TabItemCpp) {
      return TabItemCpp(child.props as TabItemProps);
    } else {
      throw new Error('LanguageTabs children must be TabItemPython or TabItemCpp components');
    }
  });

  return (
    <Tabs {...props} groupId="language">
      {tabsChildren}
    </Tabs>
  );
}

type LanguageTabItemProps = Omit<TabItemProps, 'value'>;

export function TabItemPython({ children, ...props }: LanguageTabItemProps) {
  return (
    <TabItem {...props} label="Python" value="python">
      {children}
    </TabItem>
  );
}

export function TabItemCpp({ children, ...props }: LanguageTabItemProps) {
  return (
    <TabItem {...props} label="C++" value="cpp">
      {children}
    </TabItem>
  );
}
