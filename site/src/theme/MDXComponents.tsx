import Button from '@site/src/components/Button';
import { LanguageTabs, TabItemC, TabItemCpp, TabItemPython, TabItemJS } from '@site/src/components/LanguageTabs';
import MDXComponents from '@theme-original/MDXComponents';
import TabItem from '@theme/TabItem';
import Tabs from '@theme/Tabs';

export default {
  // Reusing the default mapping
  ...MDXComponents,
  // Theme components
  Tabs,
  TabItem,
  // Custom components
  Button,
  LanguageTabs,
  TabItemPython,
  TabItemCpp,
  TabItemC,
  TabItemJS,
};
