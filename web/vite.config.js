import { defineConfig } from 'vite';

// local job server (pipeline/server.py) for the upload studio
const proxy = { '/api': 'http://127.0.0.1:8765' };

export default defineConfig({
  server: { proxy },
  preview: { proxy },
  build: { rolldownOptions: { input: { main: 'index.html', upload: 'upload.html' } } },
});
