/** Parity of the engine with the Python release on official instances: the shared core (no node or DOM APIs). */
import { Engine, type ModelConfig } from './engine';
import { parseSafetensors } from './safetensors';
import { instanceArrays, modelInputs, type Arrays } from './inputs';
import { registeredRead } from './run';
import { registeredReadAsync, type Runner } from './runner';

export interface ParityResult {
	name: string;
	split: string;
	start: number;
	instances: number;
	entries: number;
	differ: number;
	loopsDiffer: number;
	bad: number[];
	seconds: number;
	engine?: string;
}

/** cfg, the safetensors bytes, the reference predictions (reference.json) and the decoded split. */
export function parityCore(name: string, cfg: ModelConfig, weights: ArrayBuffer, ref: Record<string, unknown[]>, d: Arrays, split: string, limit: number, start = 0): ParityResult {
	const t0 = Date.now();
	const engine = new Engine(parseSafetensors(weights), cfg);
	const end = Math.min(start + limit, d.n.data.length);
	const count = end - start;
	let differ = 0;
	let loopsDiffer = 0;
	let entries = 0;
	const bad: number[] = [];
	for (let i = start; i < end; i++) {
		const { n, arrays } = instanceArrays(d, i);
		const r = registeredRead(engine, modelInputs(cfg.model, n, arrays));
		let wrong = false;
		for (const { name: out } of cfg.outputs) {
			const want = ref[`${split}/read/${out}`][i];
			const two = Array.isArray(want) && Array.isArray(want[0]);
			const M = Array.isArray(want) ? want.length : 1;
			const flat = (Array.isArray(want) ? want.flat(2) : [want]) as number[];
			const got = r.pred[out];
			for (let k = 0; k < got.length; k++) {
				const idx = two ? Math.floor(k / n) * M + (k % n) : k;
				entries++;
				if (got[k] !== flat[idx]) {
					differ++;
					wrong = true;
				}
			}
		}
		const wantLoops = ref[`${split}/loops`] as number[] | undefined;
		if (wantLoops && wantLoops[i] !== r.loops) {
			loopsDiffer++;
			wrong = true;
		}
		if (wrong) bad.push(i);
	}
	return { name, split, start, instances: count, entries, differ, loopsDiffer, bad, seconds: (Date.now() - t0) / 1000 };
}

/** The same check with a runner (CPU or GPU engine). */
export async function parityRunner(name: string, cfg: ModelConfig, runner: Runner, ref: Record<string, unknown[]>, d: Arrays, split: string, limit: number, start = 0): Promise<ParityResult> {
	const t0 = Date.now();
	const end = Math.min(start + limit, d.n.data.length);
	const count = end - start;
	let differ = 0;
	let loopsDiffer = 0;
	let entries = 0;
	const bad: number[] = [];
	for (let i = start; i < end; i++) {
		const { n, arrays } = instanceArrays(d, i);
		const r = await registeredReadAsync(runner, modelInputs(cfg.model, n, arrays));
		let wrong = false;
		for (const { name: out } of cfg.outputs) {
			const want = ref[`${split}/read/${out}`][i];
			const two = Array.isArray(want) && Array.isArray(want[0]);
			const M = Array.isArray(want) ? want.length : 1;
			const flat = (Array.isArray(want) ? want.flat(2) : [want]) as number[];
			const got = r.pred[out];
			for (let k = 0; k < got.length; k++) {
				const idx = two ? Math.floor(k / n) * M + (k % n) : k;
				entries++;
				if (got[k] !== flat[idx]) {
					differ++;
					wrong = true;
				}
			}
		}
		const wantLoops = ref[`${split}/loops`] as number[] | undefined;
		if (wantLoops && wantLoops[i] !== r.loops) {
			loopsDiffer++;
			wrong = true;
		}
		if (wrong) bad.push(i);
	}
	return { name, split, start, instances: count, entries, differ, loopsDiffer, bad, seconds: (Date.now() - t0) / 1000, engine: runner.kind };
}
