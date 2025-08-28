import type { ClassValue } from 'clsx';
import { clsx } from 'clsx';
import ColorHash from 'color-hash';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// export function toJson<T>(obj: T): T {
//   return JSON.parse(JSON.stringify(obj));
// }
export const toJson = <T>(obj: T): T => {
  return JSON.parse(JSON.stringify(obj));
};

export const objectKeys = <T extends object>(obj?: T): Array<keyof T> => {
  if (obj === undefined) return [];
  return Object.keys(obj) as Array<keyof T>;
};

export const colorHash = new ColorHash();

export type PageParams = {
  page?: number | string | null;
  pageSize?: number | string | null;
};

export const parsePageParams = ({ page = 1, pageSize = 20 }: PageParams) => {
  const p = isNaN(parseInt(String(page))) ? 1 : parseInt(String(page));
  const ps = isNaN(parseInt(String(pageSize)))
    ? 20
    : parseInt(String(pageSize));

  page = Math.max(1, Math.floor(p)) || 1;
  pageSize = Math.max(1, Math.min(Math.floor(ps), 50)) || 20;

  return {
    page,
    pageSize,
  };
};
