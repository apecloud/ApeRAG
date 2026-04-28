import fs from 'fs';
import path from 'path';

const I18N_DIR = path.resolve(process.cwd(), 'src/i18n');

const readJson = (filepath) => {
  try {
    return JSON.parse(fs.readFileSync(filepath, 'utf8'));
  } catch (error) {
    throw new Error(`Invalid JSON in ${filepath}: ${error.message}`);
  }
};

const stableStringify = (value) => `${JSON.stringify(value, null, 2)}\n`;

const flattenKeys = (value, prefix = '') => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return [prefix];
  }

  return Object.keys(value)
    .sort()
    .flatMap((key) =>
      flattenKeys(value[key], prefix ? `${prefix}.${key}` : key),
    );
};

const localeDirs = fs
  .readdirSync(I18N_DIR)
  .map((name) => path.join(I18N_DIR, name))
  .filter((filepath) => fs.statSync(filepath).isDirectory())
  .sort();

const locales = localeDirs.map((filepath) => path.basename(filepath));
const namespaceNames = new Set();
const localeMessages = new Map();

for (const locale of locales) {
  const localeDir = path.join(I18N_DIR, locale);
  const messages = {};

  for (const name of fs.readdirSync(localeDir).sort()) {
    if (!name.endsWith('.json')) continue;
    const namespace = name.replace(/\.json$/, '');
    namespaceNames.add(namespace);
    messages[namespace] = readJson(path.join(localeDir, name));
  }

  localeMessages.set(locale, messages);
}

const failures = [];

for (const namespace of [...namespaceNames].sort()) {
  const baselineLocale = locales.find(
    (locale) => localeMessages.get(locale)?.[namespace],
  );
  const baselineKeys = new Set(
    flattenKeys(localeMessages.get(baselineLocale)[namespace]),
  );

  for (const locale of locales) {
    const namespaceMessages = localeMessages.get(locale)?.[namespace];
    if (!namespaceMessages) {
      failures.push(`${locale}/${namespace}.json is missing`);
      continue;
    }

    const keys = new Set(flattenKeys(namespaceMessages));
    for (const key of baselineKeys) {
      if (!keys.has(key))
        failures.push(`${locale}/${namespace}.json missing ${key}`);
    }
    for (const key of keys) {
      if (!baselineKeys.has(key)) {
        failures.push(`${locale}/${namespace}.json has extra ${key}`);
      }
    }
  }
}

for (const locale of locales) {
  const aggregatePath = path.join(I18N_DIR, `${locale}.json`);
  const expected = stableStringify(localeMessages.get(locale));
  const actual = fs.existsSync(aggregatePath)
    ? fs.readFileSync(aggregatePath, 'utf8')
    : '';

  if (actual !== expected) {
    failures.push(`${locale}.json is out of sync; run yarn i18n:sync`);
  }
}

if (failures.length > 0) {
  console.error('i18n check failed:');
  for (const failure of failures) {
    console.error(`- ${failure}`);
  }
  process.exit(1);
}

console.log(`i18n check passed for ${locales.length} locales`);
