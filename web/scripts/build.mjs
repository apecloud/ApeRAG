import fs from 'fs';
import path from 'path';

const ROOT_DIR = process.cwd();
const BUILD_DIR = path.join(ROOT_DIR, 'build');

const readme = `# Getting Started

Requirements:

Install Node.js version >= 20.16, which can be checked by running node -v.


# deploy with node

## start
\`\`\`
nohup node /path/to/server.js > access.log 2>&1 &
\`\`\`

## stop
\`\`\`
lsof -t -i:3000 | xargs kill
\`\`\`


# deploy with pm2

\`\`\`
yarn global add pm2
\`\`\`


## start
\`\`\`
pm2 start /path/to/server.js
\`\`\`

## logs
\`\`\`
pm2 list
pm2 logs
\`\`\`

## stop
\`\`\`
pm2 stop server
\`\`\`

`;

/**
 * copy build files
 * @param {string} src
 * @param {string} dest
 */
const copyDir = (src, dest) => {
  const copy = (copySrc, copyDest) => {
    const list = fs.readdirSync(copySrc);
    list.forEach((item) => {
      const ss = path.resolve(copySrc, item);
      const stat = fs.statSync(ss);
      const curSrc = path.resolve(copySrc, item);
      const curDest = path.resolve(copyDest, item);

      if (stat.isFile()) {
        fs.createReadStream(curSrc).pipe(fs.createWriteStream(curDest));
      } else if (stat.isDirectory()) {
        fs.mkdirSync(curDest, { recursive: true });
        copy(curSrc, curDest);
      }
    });
  };

  if (!fs.existsSync(dest)) {
    fs.mkdirSync(dest);
  }
  copy(src, dest);
};

const deleteDir = function (dir) {
  if (fs.existsSync(dir)) {
    fs.readdirSync(dir).forEach((file) => {
      const curPath = path.join(dir, file);
      if (fs.lstatSync(curPath).isDirectory()) {
        deleteDir(curPath);
      } else {
        fs.unlinkSync(curPath);
      }
    });
    fs.rmdirSync(dir);
  }
};

const deploy = () => {
  deleteDir(BUILD_DIR);
  fs.mkdirSync(BUILD_DIR);
  copyDir(path.join(ROOT_DIR, './.next/standalone'), BUILD_DIR);
  copyDir(
    path.join(ROOT_DIR, './.next/static'),
    path.join(BUILD_DIR, './.next/static'),
  );
  copyDir(path.join(ROOT_DIR, './public'), path.join(BUILD_DIR, './public'));
  fs.writeFileSync(path.join(BUILD_DIR, 'readme.md'), readme);
};

deploy();
