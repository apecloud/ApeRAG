import fs from 'fs';
import _ from 'lodash';
import path from 'path';

export const ROOT_DIR = process.cwd();
export const DOCS_DIR = path.join(ROOT_DIR, 'docs');

const getUrlName = (str: string) => {
  return getUrlPath(str)
    .split('/')
    .filter((s) => s !== '')
    .join('-');
};

const getUrlPath = (str: string) => {
  return trimSurfix(str.replace(ROOT_DIR, ''));
};

const trimSurfix = (str: string) => {
  return str.replace(/\.mdx/, '');
};

const getTitle = (str: string) => {
  return _.startCase(trimSurfix(str));
};

export type DocsSideBar = {
  name: string;
  title: string;
  href?: string;
  type: 'group' | 'folder' | 'file';
  children?: DocsSideBar[];
};

export const getDocsSideBar = (dir = DOCS_DIR): DocsSideBar[] => {
  const dirs = fs.readdirSync(dir);

  return dirs.map((group) => {
    const p = path.join(dir, group);
    const stat = fs.statSync(p);
    const items = {
      name: getUrlName(p),
      title: getTitle(group),
    };
    if (stat.isDirectory()) {
      const children = getDocsSideBar(p);
      return {
        ...items,
        type:
          dir === DOCS_DIR
            ? 'group'
            : children.some((child) => child.type === 'folder')
              ? 'group'
              : 'folder',
        children,
      };
    } else {
      return {
        ...items,
        type: 'file',
        href: getUrlPath(p),
      };
    }
  });
};
