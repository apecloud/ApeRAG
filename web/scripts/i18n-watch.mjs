import chokidar from 'chokidar';
import fs from 'fs';
import path from 'path';

/**
 * input dir
 */
const I18N_DIR = path.resolve(process.cwd(), 'src/i18n');

/**
 * get watch folders and clear i18n messages
 */
const watchFolder = [];

fs.readdirSync(I18N_DIR).forEach((name) => {
  const f = path.join(I18N_DIR, name);
  const stat = fs.statSync(f);
  if (stat.isDirectory()) {
    watchFolder.push(f);
  }
  if (stat.isFile() && f.endsWith('.json')) {
    fs.rmSync(f);
  }
});

/**
 * watcher instance
 */
const watcher = chokidar.watch(watchFolder, {
  persistent: true,

  ignored: (path, stats) =>
    Boolean(stats?.isFile()) && !new RegExp(/.*\.json$/).test(path),
});

/**
 * set locale messages
 */
const onFileChange = async (filepath) => {
  const [locale, name] = filepath
    .replace(`${I18N_DIR}/`, '')
    .replace(/\.json$/, '')
    .split('/');
  const localeFile = path.resolve(I18N_DIR, `${locale}.json`);

  if (!fs.existsSync(localeFile)) {
    fs.writeFileSync(localeFile, '{}');
  }

  const localeMsg = JSON.parse(String(fs.readFileSync(localeFile)));

  let msg;
  if (fs.existsSync(filepath)) {
    try {
      msg = JSON.parse(String(fs.readFileSync(filepath)));
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
    } catch (err) {
      console.log(`error: ${filepath}`);
      msg = {};
    }
    localeMsg[name] = msg;
  } else {
    delete localeMsg[name];
  }

  fs.writeFileSync(localeFile, JSON.stringify(localeMsg, null, 2) + '\n');
};

/**
 * watcher events
 */
watcher.on('change', async (filepath) => {
  console.log(`${filepath.replace(I18N_DIR, '')} has been changed\n`);
  await onFileChange(filepath);
});
watcher.on('add', async (filepath) => {
  console.log(`${filepath.replace(I18N_DIR, '')} has been added\n`);
  await onFileChange(filepath);
});
watcher.on('unlink', async (filepath) => {
  console.log(`${filepath.replace(I18N_DIR, '')} has been removed\n`);
  await onFileChange(filepath);
});

console.log('Started watching i18n files...\n');
