import { getLocale } from '@/services/cookies';
import fs from 'fs';
import path from 'path';

const baseDir = path.join(process.cwd(), 'src/lib/prompt-template');

export const getDefaultPrompt = async () => {
  const locale = await getLocale();
  if (locale === 'zh-CN') {
    return {
      query: String(
        fs.readFileSync(path.join(baseDir, 'query-prompt.zh-CN.md')),
      ),
      system: String(
        fs.readFileSync(path.join(baseDir, 'system-prompt.zh-CN.md')),
      ),
    };
  } else {
    return {
      query: String(
        fs.readFileSync(path.join(baseDir, 'query-prompt.en-US.md')),
      ),
      system: String(
        fs.readFileSync(path.join(baseDir, 'system-prompt.en-US.md')),
      ),
    };
  }
};
