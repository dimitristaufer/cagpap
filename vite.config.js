import fs from 'node:fs';
import path from 'node:path';
import { defineConfig } from 'vite';

const MIME_BY_EXT = new Map([
  ['.bin', 'application/octet-stream'],
  ['.css', 'text/css; charset=utf-8'],
  ['.html', 'text/html; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.mjs', 'text/javascript; charset=utf-8'],
  ['.onnx', 'application/octet-stream'],
  ['.onnx_data', 'application/octet-stream'],
  ['.txt', 'text/plain; charset=utf-8'],
  ['.wasm', 'application/wasm'],
  ['.woff2', 'font/woff2'],
]);

function contentTypeFor(filePath) {
  return MIME_BY_EXT.get(path.extname(filePath)) || 'application/octet-stream';
}

function distPathForRequest(reqUrl, base = '/cagpap') {
  const parsed = new URL(reqUrl || '/', 'http://localhost');
  let pathname = decodeURIComponent(parsed.pathname);
  if (base && pathname === base) {
    pathname = '/index.html';
  } else if (base && pathname.startsWith(`${base}/`)) {
    pathname = pathname.slice(base.length);
  }
  if (pathname === '/') pathname = '/index.html';
  const normalized = pathname.replace(/^\/+/, '');
  return path.resolve(process.cwd(), 'dist', normalized);
}

function configureBrotliPreview(server) {
  server.middlewares.use((req, res, next) => {
    const method = String(req.method || 'GET').toUpperCase();
    if (method !== 'GET' && method !== 'HEAD') return next();
    if (!String(req.headers['accept-encoding'] || '').includes('br')) return next();

    const requestedPath = distPathForRequest(req.url);
    const brPath = `${requestedPath}.br`;
    if (!fs.existsSync(requestedPath) || !fs.existsSync(brPath)) return next();

    const stat = fs.statSync(brPath);
    res.statusCode = 200;
    res.setHeader('Content-Encoding', 'br');
    res.setHeader('Content-Type', contentTypeFor(requestedPath));
    res.setHeader('Content-Length', String(stat.size));
    res.setHeader('Vary', 'Accept-Encoding');
    if (method === 'HEAD') {
      res.end();
      return;
    }
    fs.createReadStream(brPath).pipe(res);
  });
}

export default defineConfig({
  base: '/cagpap/',
  plugins: [
    {
      name: 'brotli-preview',
      configurePreviewServer: configureBrotliPreview,
    },
    {
      name: 'dev-profile-fetch-proxy',
      configureServer(server) {
        server.middlewares.use('/__fetch', async (req, res) => {
          try {
            const requestUrl = new URL(req.url || '', 'http://localhost');
            const target = requestUrl.searchParams.get('url');
            if (!target) {
              res.statusCode = 400;
              res.setHeader('content-type', 'application/json; charset=utf-8');
              res.end(JSON.stringify({ error: 'Missing url query parameter.' }));
              return;
            }

            let parsedTarget;
            try {
              parsedTarget = new URL(target);
            } catch {
              res.statusCode = 400;
              res.setHeader('content-type', 'application/json; charset=utf-8');
              res.end(JSON.stringify({ error: 'Invalid target url.' }));
              return;
            }

            if (!['http:', 'https:'].includes(parsedTarget.protocol)) {
              res.statusCode = 400;
              res.setHeader('content-type', 'application/json; charset=utf-8');
              res.end(JSON.stringify({ error: 'Only http/https targets are allowed.' }));
              return;
            }

            const host = parsedTarget.hostname.toLowerCase();
            const allowed =
              host.includes('scholar.google.') ||
              host === 'researchgate.net' ||
              host.endsWith('.researchgate.net') ||
              host === 'r.jina.ai';

            if (!allowed) {
              res.statusCode = 403;
              res.setHeader('content-type', 'application/json; charset=utf-8');
              res.end(JSON.stringify({ error: 'Host not allowed for dev proxy.' }));
              return;
            }

            const upstream = await fetch(parsedTarget.toString(), {
              method: 'GET',
              redirect: 'follow',
              headers: {
                'user-agent':
                  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                'accept-language': 'en-US,en;q=0.9',
              },
            });

            const body = await upstream.text();
            res.statusCode = upstream.status;
            res.setHeader('content-type', 'text/plain; charset=utf-8');
            res.end(body);
          } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            res.statusCode = 502;
            res.setHeader('content-type', 'application/json; charset=utf-8');
            res.end(JSON.stringify({ error: `Dev proxy request failed: ${message}` }));
          }
        });
      },
    },
  ],
});
