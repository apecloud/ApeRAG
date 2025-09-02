import { getLocale } from '@/services/cookies';
import fs from 'fs';
import grayMatter from 'gray-matter';
import _ from 'lodash';
import path from 'path';
import readYamlFile from 'read-yaml-file';

const ROOT_DIR = process.cwd();
export const DOCS_DIR = path.join(ROOT_DIR, 'docs');

const getId = (pathname: string) => {
  const href = getHref(pathname);
  return href
    .split('/')
    .filter((s) => s !== '')
    .join('-');
};

const getHref = (pathname: string) => {
  const url = trimSurfix(pathname.replace(ROOT_DIR, ''));
  return url.replace(/\/(en-US|zh-CN)/, '');
};

const trimSurfix = (str: string) => {
  return str.replace(/\.mdx?$/, '');
};

const getTitleByFilename = (str: string) => {
  return _.startCase(trimSurfix(str));
};

export type DocsSideBar = {
  id: string;
  title: string;
  type: 'group' | 'folder' | 'file';

  description?: string;
  href?: string;
  children?: DocsSideBar[];
  collapsed?: boolean;
};

type FileMetadata = {
  title?: string;
  description?: string;
  keywords?: string;
};
const readFile = async (filepath: string, folder: string) => {
  const isExists = fs.existsSync(filepath);

  let metadata: FileMetadata = {};

  if (isExists) {
    const { data } = grayMatter(fs.readFileSync(filepath, 'utf8'));
    metadata = data as FileMetadata;
  }

  const item: DocsSideBar = {
    id: getId(filepath),
    title: metadata.title || getTitleByFilename(folder),
    description: metadata.description,
    type: 'file',
    href: getHref(filepath),
  };
  return item;
};

type FolderMetadata = {
  title?: string;
  position?: number;
  collapsed?: boolean;
};
const readFolder = async (folderDir: string, parent: string) => {
  const children = await getDocsSideBar(folderDir);
  const metadataFilePath = path.join(folderDir, '_category.yaml');
  let metadata: FolderMetadata = {
    position: 0,
    collapsed: true,
  };

  if (fs.existsSync(metadataFilePath)) {
    metadata = {
      ...metadata,
      ...(await readYamlFile<FolderMetadata>(metadataFilePath)),
    };
  }

  const parentIsDocsRoot =
    folderDir.replace(DOCS_DIR, '').split('/').length === 3;
  const childrenIsParent = children.some((child) => child.type === 'folder');

  const item: DocsSideBar = {
    id: getId(folderDir),
    title: metadata.title || getTitleByFilename(parent),
    type: parentIsDocsRoot || childrenIsParent ? 'group' : 'folder',
    children,
  };

  return item;
};

export async function getDocsSideBar(dir?: string): Promise<DocsSideBar[]> {
  const locale = await getLocale();
  const baseDir = dir || path.join(DOCS_DIR, locale);
  const dirs = fs.readdirSync(baseDir);
  const result = await Promise.all(
    dirs.map(async (parent) => {
      const pathname = path.join(baseDir, parent);
      const stat = fs.statSync(pathname);
      if (stat.isDirectory()) {
        return await readFolder(pathname, parent);
      }

      // should be a mdx file
      if (pathname.match(/\.mdx?$/) === null) {
        return;
      }

      return await readFile(pathname, parent);
    }),
  );

  return result.filter((item) => item !== undefined);
}

export async function getIndexPageUrl(
  dir?: string,
): Promise<string | undefined> {
  const locale = await getLocale();
  const baseDir = dir || path.join(DOCS_DIR, locale);
  const dirs = fs.readdirSync(baseDir);
  let url: string | undefined;

  for (const i in dirs) {
    const pathname = path.join(baseDir, dirs[i]);
    const stat = fs.statSync(pathname);

    if (stat.isFile() && pathname.match(/\.mdx?$/)) {
      url = getHref(pathname);
    } else if (stat.isDirectory()) {
      url = await getIndexPageUrl(pathname);
    }
    if (url) {
      break;
    }
  }
  return url;
}
