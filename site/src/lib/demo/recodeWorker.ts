/**
 * The recoding test, off the page's thread: two models of one task (a compare-first block and a block that embeds the
 * positions) each run by the registered rule on an instance and on the same instance with its positions recoded.
 */
import type { ModelConfig } from '../model/engine';
import { modelInputs, type Arrays } from '../model/inputs';
import { parseSafetensors } from '../model/safetensors';
import { makeRunner, registeredReadAsync, type Runner } from '../model/runner';

export type RecodeRequest = { id: number; base: string; names: string[]; n: number; original: Arrays; recoded: Arrays; engine: 'gpu' | 'cpu' };

export interface RecodeResult {
	name: string;
	block: string;
	original: Record<string, Int32Array>;
	recoded: Record<string, Int32Array>;
	loops: [number, number];
}

const runners = new Map<string, { r: Runner; cfg: ModelConfig }>();

async function load(base: string, name: string, engine: 'gpu' | 'cpu') {
	const key = `${engine}:${name}`;
	if (!runners.has(key)) {
		const [cfg, buf] = await Promise.all([
			fetch(`${base}/models/${name}/config.json`).then((r) => r.json() as Promise<ModelConfig>),
			fetch(`${base}/models/${name}/model.safetensors`).then((r) => r.arrayBuffer())
		]);
		runners.set(key, { r: await makeRunner(parseSafetensors(buf), cfg, engine), cfg });
	}
	return runners.get(key)!;
}

self.onmessage = async (ev: MessageEvent<RecodeRequest>) => {
	const q = ev.data;
	const post = (m: unknown) => (self as unknown as Worker).postMessage(m);
	try {
		const out: RecodeResult[] = [];
		for (const name of q.names) {
			const { r, cfg } = await load(q.base, name, q.engine);
			post({ id: q.id, progress: `${name}: the graph as given` });
			const a = await registeredReadAsync(r, modelInputs(cfg.model, q.n, q.original));
			post({ id: q.id, progress: `${name}: positions recoded` });
			const b = await registeredReadAsync(r, modelInputs(cfg.model, q.n, q.recoded));
			out.push({ name, block: cfg.model.block, original: a.pred, recoded: b.pred, loops: [a.loops, b.loops] });
		}
		post({ id: q.id, results: out });
	} catch (err) {
		post({ id: q.id, error: String(err) });
	}
};
