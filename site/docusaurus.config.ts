import type * as Preset from '@docusaurus/preset-classic';
import type { Config } from '@docusaurus/types';
import { themes as prismThemes } from 'prism-react-renderer';
import GenAISamplesDocsPlugin from './src/plugins/genai-samples-docs-plugin';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

// GITHUB_REPOSITORY env var is set by GitHub Actions
const GITHUB_REPOSITORY = process.env.GITHUB_REPOSITORY || 'openvinotoolkit/openvino.genai';
const [organizationName, projectName] = GITHUB_REPOSITORY.split('/');

const config: Config = {
  title: 'OpenVINO GenAI',
  favicon: 'img/favicon.png',

  // Set the production url of your site here
  url: `https://${organizationName}.github.io`,
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: `/${projectName}/`,

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName, // Usually your GitHub org/user name.
  projectName, // Usually your repo name.

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'throw',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          async sidebarItemsGenerator({ defaultSidebarItemsGenerator, ...args }) {
            const excludeCategories = args.item.customProps?.excludeCategories as
              | string[]
              | undefined;

            const sidebarItems = await defaultSidebarItemsGenerator(args);

            return sidebarItems.filter((item) => {
              if (excludeCategories && item.type === 'category') {
                return !excludeCategories.includes(item.label);
              }
              return true;
            });
          },
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl: `https://github.com/${organizationName}/${projectName}/tree/master/site`,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    colorMode: {
      disableSwitch: true,
      defaultMode: 'light',
    },
    navbar: {
      title: 'OpenVINO GenAI',
      logo: {
        alt: 'Intel logo',
        src: 'img/intel-logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'genaiDocsSidebar',
          position: 'left',
          label: 'Documentation',
          to: '/docs',
        },
        {
          href: 'https://github.com/openvinotoolkit/openvino.genai',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'OpenVINO',
          items: [
            {
              label: 'OpenVINO™ Documentation',
              href: 'https://docs.openvino.ai/',
            },
            {
              label: 'Case Studies',
              href: 'https://www.intel.com/content/www/us/en/internet-of-things/ai-in-production/success-stories.html',
            },
          ],
        },
        {
          title: 'Legal',
          items: [
            {
              label: 'Terms of Use',
              href: 'https://docs.openvino.ai/2025/about-openvino/additional-resources/terms-of-use.html',
            },
            {
              label: 'Responsible AI',
              href: 'https://www.intel.com/content/www/us/en/artificial-intelligence/responsible-ai.html',
            },
          ],
        },
        {
          title: 'Privacy',
          items: [
            {
              label: 'Cookies',
              href: 'https://www.intel.com/content/www/us/en/privacy/intel-cookie-notice.html',
            },
            {
              label: 'Privacy',
              href: 'https://www.intel.com/content/www/us/en/privacy/intel-privacy-notice.html',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Intel Corporation
                Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries.
                Other names and brands may be claimed as the property of others.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,

  themes: [
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        hashed: true,
        highlightSearchTermsOnTargetPage: true,
        searchBarShortcutHint: false,
      },
    ],
  ],
  plugins: [GenAISamplesDocsPlugin],
};

export default config;
