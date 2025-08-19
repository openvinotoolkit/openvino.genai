import globals from "globals";
import eslint from "@eslint/js";
import prettierConfig from "eslint-plugin-prettier/recommended";
import tseslint from "typescript-eslint";

export default [
  {
    ignores: ["types/", "dist/"],
  },
  eslint.configs.recommended,
  {
    files: ["**/*.{js,mjs,cjs,ts}"],
    languageOptions: {
      globals: globals.node,
      parser: tseslint.parser,
    },
  },
  {
    rules: {
      "no-var": ["error"],
      camelcase: ["error"],
      "prefer-destructuring": ["error", { object: true, array: false }],
    },
  },
  prettierConfig,
] satisfies tseslint.ConfigArray;
