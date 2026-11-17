/**
 * Parity: the page's engine predicts what the Python release predicts, on the official instances.
 * For every model in static/models/index.json: the registered read on the first PARITY_LIMIT instances (default 4) of
 * PARITY_SPLIT (default val) against the reference predictions the model made when it was trained.
 * Full runs at 64 items are slow in one thread (seconds per instance); scripts/parity-cli.ts runs them in parallel.
 * PARITY_FILTER restricts the models by name.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, test } from 'vitest';
import { parity } from './parity';

const STATIC = join(import.meta.dirname, '../../../static');
const SPLIT = process.env.PARITY_SPLIT ?? 'val';
const LIMIT = Number(process.env.PARITY_LIMIT ?? 4);
const FILTER = process.env.PARITY_FILTER ?? '';
const index = JSON.parse(readFileSync(join(STATIC, 'models/index.json'), 'utf8')) as { name: string; task: string }[];

describe('engine parity with the Python release', () => {
	for (const m of index.filter((q) => q.name.includes(FILTER))) {
		test(m.name, { timeout: 3_600_000 }, () => {
			const r = parity(STATIC, m.name, m.task, SPLIT, LIMIT);
			expect({ differ: r.differ, loopsDiffer: r.loopsDiffer }).toEqual({ differ: 0, loopsDiffer: 0 });
		});
	}
});
