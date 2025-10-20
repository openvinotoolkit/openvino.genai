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
      "@typescript-eslint/no-namespace": ["error", { allowDeclarations: true }],
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
            "max_num_batched_tokens",
            "num_kv_blocks",
            "cache_size",
            "dynamic_split_fuse",
            "pooling_type",
            "json_schema",
            "structured_output_config",
            "structural_tags_config",
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
