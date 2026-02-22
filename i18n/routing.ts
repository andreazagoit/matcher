/**
 * next-intl locale config — single locale, no URL prefix.
 * To add more languages in the future, add them to `locales`
 * and create the corresponding messages/<locale>.json file.
 */
export const defaultLocale = "it" as const;
export type Locale = typeof defaultLocale;
