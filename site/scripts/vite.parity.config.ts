// Bundle scripts/parity-cli.ts into one node script: npx vite build --config scripts/vite.parity.config.ts
import { defineConfig } from 'vite';
export default defineConfig({
	build: {
		ssr: 'scripts/parity-cli.ts',
		outDir: 'dist-parity',
		emptyOutDir: true,
		rollupOptions: { output: { entryFileNames: 'parity.mjs' } }
	}
});
