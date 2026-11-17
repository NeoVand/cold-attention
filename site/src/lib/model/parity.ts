/** Parity of the page's engine with the Python release on official instances, reading the static files from disk (node). */
import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import type { ModelConfig } from './engine';
import { decodeArrays, type SplitJSON } from './inputs';
import { parityCore, type ParityResult } from './paritycore';

export type { ParityResult };

export function parity(staticDir: string, name: string, task: string, split: string, limit: number, start = 0): ParityResult {
	const dir = join(staticDir, 'models', name);
	const cfg = JSON.parse(readFileSync(join(dir, 'config.json'), 'utf8')) as ModelConfig;
	const buf = readFileSync(join(dir, 'model.safetensors'));
	const ref = JSON.parse(readFileSync(join(dir, 'reference.json'), 'utf8')) as Record<string, unknown[]>;
	const file = join(staticDir, 'instances', task, `${split}.json`);
	if (!existsSync(file)) throw new Error(`missing ${file}`);
	const d = decodeArrays(JSON.parse(readFileSync(file, 'utf8')) as SplitJSON);
	return parityCore(name, cfg, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), ref, d, split, limit, start);
}
