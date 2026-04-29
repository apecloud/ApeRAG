import { getLocale } from '@/services/cookies';
import fs from 'fs';
import _ from 'lodash';
import path from 'path';
import { parse as parseYaml } from 'yaml';

const ROOT_DIR = process.cwd();
const resolveDocsDir = () => {
  const envDocsDir = process.env.APERAG_DOCS_DIR;
  const candidates = [
    envDocsDir,
    path.join(ROOT_DIR, 'docs'),
    path.join(ROOT_DIR, '..', 'docs'),
  ].filter((candidate): candidate is string => Boolean(candidate));

  return (
    candidates.find((candidate) => fs.existsSync(candidate)) ?? candidates[0]
  );
};

export const DOCS_DIR = resolveDocsDir();

const getId = (pathname: string) => {
  const href = getHref(pathname);
  return href
    .split('/')
    .filter((s) => s !== '')
    .join('-');
};

const getHref = (pathname: string) => {
  const relativePath = path
    .relative(DOCS_DIR, pathname)
    .split(path.sep)
    .join('/');
  const routePath = relativePath.replace(/^(en-US|zh-CN)(\/|$)/, '');

  return `/docs/${trimSurfix(routePath)}`.replace(/\/+/g, '/');
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
  position?: number;
};

type FileMetadata = {
  title?: string;
  description?: string;
  keywords?: string;
  position?: number;
};

const FRONTMATTER_PATTERN = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/;

const readFrontmatter = <T>(content: string): Partial<T> => {
  const match = FRONTMATTER_PATTERN.exec(content);
  if (!match) {
    return {};
  }

  return (parseYaml(match[1]) as Partial<T> | null) ?? {};
};

const readFile = async (filepath: string, folder: string) => {
  const isExists = fs.existsSync(filepath);

  let metadata: FileMetadata = {};

  if (isExists) {
    metadata = readFrontmatter<FileMetadata>(fs.readFileSync(filepath, 'utf8'));
  }

  const item: DocsSideBar = {
    id: getId(filepath),
    title: metadata.title || getTitleByFilename(folder),
    description: metadata.description,
    type: 'file',
    href: getHref(filepath),
    position: metadata.position ?? 999,
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
      ...(parseYaml(
        fs.readFileSync(metadataFilePath, 'utf8'),
      ) as FolderMetadata),
    };
  }

  const relativeFolderParts = path
    .relative(DOCS_DIR, folderDir)
    .split(path.sep)
    .filter(Boolean);
  const parentIsDocsRoot = relativeFolderParts.length === 2;
  const childrenIsParent = children.some((child) => child.type === 'folder');

  const item: DocsSideBar = {
    id: getId(folderDir),
    title: metadata.title || getTitleByFilename(parent),
    type: parentIsDocsRoot || childrenIsParent ? 'group' : 'folder',
    children,
    position: metadata.position ?? 999,
  };

  return item;
};

export async function getDocsSideBar(dir?: string): Promise<DocsSideBar[]> {
  const locale = await getLocale();
  const baseDir = dir || path.join(DOCS_DIR, locale);
  const dirs = fs.readdirSync(baseDir);
  const result = await Promise.all(
    dirs.map(async (parent) => {
      // Skip images directory from sidebar
      if (parent === 'images') {
        return;
      }

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

  return result
    .filter((item) => item !== undefined)
    .sort((a, b) => (a.position ?? 999) - (b.position ?? 999));
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
