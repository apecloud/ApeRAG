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
