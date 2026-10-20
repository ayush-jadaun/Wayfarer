import { defineConfig } from 'vite';

// COOP/COEP enable SharedArrayBuffer, which the splat sorter uses for faster sorting.
const headers = { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' };
// local job server (pipeline/server.py) for the upload studio
const proxy = { '/api': 'http://127.0.0.1:8765' };

export default defineConfig({
  server: { headers, proxy },
  preview: { headers, proxy },
  build: { rolldownOptions: { input: { main: 'index.html', upload: 'upload.html' } } },
});
