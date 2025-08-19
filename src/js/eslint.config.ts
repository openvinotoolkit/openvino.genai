import globals from "globals";
import eslint from "@eslint/js";
import prettierConfig from "eslint-plugin-prettier/recommended";
import tseslint from "typescript-eslint";

export default tseslint.config(
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
  ...tseslint.configs.recommended.map((config) => ({
    ...config,
    files: ["**/*.ts"],
  })),
  {
    rules: {
      "no-var": ["error"],
      "prefer-destructuring": ["error", { object: true, array: false }],
      "@typescript-eslint/no-misused-new": "off",
      "@typescript-eslint/no-explicit-any": "off",
    },
  },
  prettierConfig,
);
