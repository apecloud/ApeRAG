#!/usr/bin/env node
// Copy ``pdfjs-dist`` runtime assets (``cmaps`` + ``standard_fonts``)
// from ``node_modules/pdfjs-dist/`` into ``web/public/`` so Next.js
// serves them as static assets at runtime.
//
// Why this exists: react-pdf delegates to PDF.js, which needs CMap
// (character map) files to render non-Latin glyphs (CJK, Cyrillic,
// Arabic, etc.) and "standard fonts" data to render PDFs that
// reference Adobe Type 1 base fonts (Helvetica / Times / Courier
// without embedded font data). Without these assets reachable at
// the URLs configured on ``<Document options={{ cMapUrl, ... }}>``,
// PDF.js silently skips non-Latin characters and falls back to
// generic glyphs for the base fonts — surfacing as Chinese PDFs
// rendering only ASCII punctuation / digits (earayu2 task #12 bug
// report msg=8caa73c9).
//
// Bundling locally (vs. an unpkg/jsdelivr CDN URL) keeps ApeRAG
// usable in CN / private-cloud deploys with no outbound CDN access.
//
// Idempotent: re-running overwrites the destination tree, so it's
// safe to call from postinstall and from explicit ``yarn`` script
// invocations during dev/build.

import fs from 'node:fs/promises';
import path from 'node:path';
import url from 'node:url';

const __filename = url.fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const webRoot = path.resolve(__dirname, '..');

const PDFJS_MODULE = path.join(webRoot, 'node_modules', 'pdfjs-dist');
const PUBLIC_DIR = path.join(webRoot, 'public');

const ASSETS = [
  { from: 'cmaps', to: 'cmaps' },
  { from: 'standard_fonts', to: 'standard_fonts' },
];

async function copyDirectory(src, dst) {
  await fs.mkdir(dst, { recursive: true });
  const entries = await fs.readdir(src, { withFileTypes: true });
  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const dstPath = path.join(dst, entry.name);
    if (entry.isDirectory()) {
      await copyDirectory(srcPath, dstPath);
    } else if (entry.isFile()) {
      await fs.copyFile(srcPath, dstPath);
    }
  }
}

async function main() {
  try {
    await fs.access(PDFJS_MODULE);
  } catch {
    console.warn(
      `[copy-pdf-assets] ${PDFJS_MODULE} not found; skipping ` +
        `(run ``yarn install`` first or this is a fresh clone before deps).`,
    );
    return;
  }
  for (const { from, to } of ASSETS) {
    const src = path.join(PDFJS_MODULE, from);
    const dst = path.join(PUBLIC_DIR, to);
    try {
      await fs.access(src);
    } catch {
      console.warn(`[copy-pdf-assets] source not found: ${src} (skip)`);
      continue;
    }
    await copyDirectory(src, dst);
    console.log(
      `[copy-pdf-assets] copied ${path.relative(webRoot, src)} → ${path.relative(webRoot, dst)}`,
    );
  }
}

main().catch((err) => {
  console.error('[copy-pdf-assets] failed:', err);
  process.exit(1);
});
