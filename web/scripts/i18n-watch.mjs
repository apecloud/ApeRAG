import fs from 'fs';
import path from 'path';

const I18N_DIR = path.resolve(process.cwd(), 'src/i18n');
const WATCH_MODE = process.argv.includes('--watch');

const getLocaleDirectories = () =>
  fs
    .readdirSync(I18N_DIR)
    .map((name) => path.join(I18N_DIR, name))
    .filter((filepath) => fs.statSync(filepath).isDirectory())
    .sort();

const getLocaleName = (filepath) =>
  path.relative(I18N_DIR, filepath).split(path.sep)[0];

const getAggregateLocalePath = (locale) =>
  path.join(I18N_DIR, `${locale}.json`);

const readJsonFile = (filepath) => {
  try {
    return JSON.parse(String(fs.readFileSync(filepath)));
  } catch (error) {
    console.log(`error: ${filepath}`);
    return {};
  }
};

const syncLocale = (locale) => {
  const localeDir = path.join(I18N_DIR, locale);
  const localeMessages = {};

  fs.readdirSync(localeDir)
    .filter((name) => name.endsWith('.json'))
    .sort()
    .forEach((name) => {
      const namespace = name.replace(/\.json$/, '');
      localeMessages[namespace] = readJsonFile(path.join(localeDir, name));
    });

  fs.writeFileSync(
    getAggregateLocalePath(locale),
    JSON.stringify(localeMessages, null, 2) + '\n',
  );
};

const syncAllLocales = () => {
  const locales = getLocaleDirectories().map((filepath) =>
    path.basename(filepath),
  );
  const activeAggregates = new Set(locales.map(getAggregateLocalePath));

  for (const locale of locales) {
    syncLocale(locale);
  }

  for (const name of fs.readdirSync(I18N_DIR)) {
    const filepath = path.join(I18N_DIR, name);
    if (
      fs.statSync(filepath).isFile() &&
      filepath.endsWith('.json') &&
      !activeAggregates.has(filepath)
    ) {
      fs.rmSync(filepath);
    }
  }
};

syncAllLocales();

if (!WATCH_MODE) process.exit(0);

const { default: chokidar } = await import('chokidar');
const watchFolders = getLocaleDirectories();
const watcher = chokidar.watch(watchFolders, {
  persistent: true,
  ignored: (filepath, stats) =>
    Boolean(stats?.isFile()) && !filepath.endsWith('.json'),
});

watcher.on('change', async (filepath) => {
  console.log(`${filepath.replace(I18N_DIR, '')} has been changed\n`);
  syncLocale(getLocaleName(filepath));
});
watcher.on('add', async (filepath) => {
  console.log(`${filepath.replace(I18N_DIR, '')} has been added\n`);
  syncLocale(getLocaleName(filepath));
});
watcher.on('unlink', async (filepath) => {
  console.log(`${filepath.replace(I18N_DIR, '')} has been removed\n`);
  syncLocale(getLocaleName(filepath));
});

console.log('Started watching i18n files...\n');
