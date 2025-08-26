const eslint = require("@eslint/js");
const prettierConfig = require("eslint-plugin-prettier/recommended");
const globals = require("globals");
const tseslint = require("typescript-eslint");
const { defineConfig } = require("eslint/config");
const path = require("node:path");

module.exports = defineConfig([
  {
    ignores: ["types/", "dist/"],
  },
  {
    files: ["**/*.*js"],
    languageOptions: {
      globals: globals.node,
    },
    extends: [eslint.configs.recommended],
  },
  {
    files: ["**/*.ts"],
    languageOptions: {
      globals: globals.node,
      parser: tseslint.parser,
      parserOptions: {
        projectService: true,
        project: "./tsconfig.json",
        tsconfigRootDir: path.resolve(),
      },
    },
    extends: [tseslint.configs.recommended],
    rules: {
      "@typescript-eslint/no-require-imports": 0,
      "@typescript-eslint/no-explicit-any": "off",
    },
  },
  {
    rules: {
      "no-var": ["error"],
      camelcase: [
        "error",
        {
          allow: [
            "max_new_tokens",
            "return_decoded_results",
            "stop_strings",
            "include_stop_str_in_output",
          ],
        },
      ],
      "prefer-destructuring": ["error", { object: true, array: false }],
    },
  },
  prettierConfig, // to disable stylistic rules from ESLint
  {
    files: ["**/addon.ts"],
    rules: {
      "@typescript-eslint/no-misused-new": "off",
    },
  },
]);
