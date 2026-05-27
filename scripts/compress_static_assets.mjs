import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

const appRoot = path.resolve(new URL('..', import.meta.url).pathname);
const distRoot = path.join(appRoot, 'dist');
const MIN_BYTES = 1024;
const BROTLI_EXTENSIONS = new Set([
  '.bin',
  '.css',
  '.html',
  '.js',
  '.json',
  '.mjs',
  '.onnx',
  '.onnx_data',
  '.txt',
  '.wasm',
  '.woff2',
]);

function walkFiles(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...walkFiles(fullPath));
    } else if (entry.isFile()) {
      files.push(fullPath);
    }
  }
  return files;
}

function shouldCompress(filePath) {
  if (filePath.endsWith('.br') || filePath.endsWith('.gz')) return false;
  const stat = fs.statSync(filePath);
  if (stat.size < MIN_BYTES) return false;
  return BROTLI_EXTENSIONS.has(path.extname(filePath));
}

function compressFile(filePath) {
  const input = fs.readFileSync(filePath);
  const compressed = zlib.brotliCompressSync(input, {
    params: {
      [zlib.constants.BROTLI_PARAM_QUALITY]: 11,
      [zlib.constants.BROTLI_PARAM_SIZE_HINT]: input.length,
    },
  });
  const outPath = `${filePath}.br`;
  fs.writeFileSync(outPath, compressed);
  return {
    filePath,
    outPath,
    rawBytes: input.length,
    brBytes: compressed.length,
  };
}

function formatMb(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function main() {
  if (!fs.existsSync(distRoot)) {
    throw new Error(`Missing dist directory: ${distRoot}`);
  }

  const compressed = walkFiles(distRoot)
    .filter(shouldCompress)
    .map(compressFile);

  const rawTotal = compressed.reduce((sum, item) => sum + item.rawBytes, 0);
  const brTotal = compressed.reduce((sum, item) => sum + item.brBytes, 0);
  console.log(
    `Wrote ${compressed.length} Brotli assets (${formatMb(rawTotal)} raw -> ${formatMb(brTotal)} br).`
  );
}

main();
