import MDXComponents from '@theme-original/MDXComponents'
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { LanguageTabs, TabItemPython, TabItemCpp } from '@site/src/components/LanguageTabs';

export default {
  // Reusing the default mapping
  ...MDXComponents,
  // Theme components
  Tabs,
  TabItem,
  // Custom components
  LanguageTabs,
  TabItemPython,
  TabItemCpp,
};
