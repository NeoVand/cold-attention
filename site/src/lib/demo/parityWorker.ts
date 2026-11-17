/** The parity check, in the browser: fetch a model, its reference predictions and an official split; run; report. */
import type { ModelConfig } from '../model/engine';
import { decodeArrays, type Arrays, type SplitJSON } from '../model/inputs';
import { parityRunner } from '../model/paritycore';
import { parseSafetensors } from '../model/safetensors';
import { makeRunner } from '../model/runner';

const splits = new Map<string, Arrays>();

self.onmessage = async (ev: MessageEvent<{ base: string; name: string; task: string; split: string; limit: number; engine: 'gpu' | 'cpu' }>) => {
	const { base, name, task, split, limit, engine } = ev.data;
	try {
		const key = `${task}/${split}`;
		if (!splits.has(key)) splits.set(key, decodeArrays((await fetch(`${base}/instances/${key}.json`).then((r) => r.json())) as SplitJSON));
		const [cfg, weights, ref] = await Promise.all([
			fetch(`${base}/models/${name}/config.json`).then((r) => r.json() as Promise<ModelConfig>),
			fetch(`${base}/models/${name}/model.safetensors`).then((r) => r.arrayBuffer()),
			fetch(`${base}/models/${name}/reference.json`).then((r) => r.json() as Promise<Record<string, unknown[]>>)
		]);
		const runner = await makeRunner(parseSafetensors(weights), cfg, engine);
		const result = await parityRunner(name, cfg, runner, ref, splits.get(key)!, split, limit);
		runner.destroy();
		(self as unknown as Worker).postMessage({ ok: true, result });
	} catch (e) {
		(self as unknown as Worker).postMessage({ ok: false, name, error: String(e) });
	}
};
