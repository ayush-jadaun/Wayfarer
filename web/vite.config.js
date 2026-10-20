import { defineConfig } from 'vite';

// COOP/COEP enable SharedArrayBuffer, which the splat sorter uses for faster sorting.
const headers = { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' };

export default defineConfig({ server: { headers }, preview: { headers } });
