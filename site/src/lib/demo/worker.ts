/**
 * The block runs here, off the page's thread. The page sends a model and an instance; the worker runs the loops by the
 * registered rule, posting each loop's predictions as it goes, and keeps every loop's trace to answer the page's
 * questions about a pair (its witnesses) or a channel pair (its verdicts over all pairs).
 */
import type { Inputs, ModelConfig, State, Trace } from '../model/engine';
import { parseSafetensors } from '../model/safetensors';
import { censusLoops } from '../model/run';
import { makeRunner, type Runner } from '../model/runner';

export type Request =
	| { type: 'load'; id: number; base: string; name: string; engine: 'gpu' | 'cpu' }
	| { type: 'run'; id: number; inputs: Inputs }
	| { type: 'grid'; id: number; loop: number; pair: number }
	| { type: 'pair'; id: number; loop: number; u: number; v: number }
	| { type: 'stop'; id: number };

export interface LoopMsg {
	type: 'loop';
	id: number;
	loop: number;
	changed: number;
	pred: Record<string, Int32Array>;
	ms: number;
	/** the channel pair whose verdicts vary most over the pairs at this loop */
	informative: number;
}

export interface PairInfo {
	/** count half: witnesses per channel ([h][n] 0/1), their counts, and each channel pair's leading value (registers) */
	witness: Uint8Array;
	counts: Uint16Array;
	lead: Float32Array | null;
	/** best/exists half: the best witness per channel (255 = none) */
	best: Uint8Array;
	/** the bits the pair holds after the loop ([D]) — census: the comparisons this loop produced */
	bits: Float32Array;
}

let engine: Runner | null = null;
let traces: (Trace | null)[] = [];
let states: State[] = [];
let n = 0;
let stopAt = -1;

const post = (m: unknown, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(m, transfer);

function verdictGrid(t: Trace | null, s: State, pair: number, h: number, D: number, census: boolean): Uint8Array {
	// 0 = fewer, 1 = same, 2 = more, per pair (u, v), for count channels (2 pair, 2 pair + 1): census from its counts,
	// the resting blocks from the bits they keep (register pairs compare numbers, not counts)
	const P = n * n;
	const g = new Uint8Array(P);
	for (let i = 0; i < P; i++) {
		if (census && t) {
			const a = t.counts[i * h + 2 * pair];
			const b = t.counts[i * h + 2 * pair + 1];
			g[i] = a > b ? 2 : a === b ? 1 : 0;
		} else g[i] = s.s[i * D + 2 * pair] ? 2 : s.s[i * D + 2 * pair + 1] ? 1 : 0;
	}
	return g;
}

/** the channel pair whose verdicts vary most over the pairs (the most telling one to show) */
function mostTelling(st: State, t: Trace | null, eng: Runner): number {
	const census = eng.block === 'census';
	if (census && !t) return 0;
	let best = 0;
	let bestH = -1;
	for (let c = 0; c < eng.h / 2; c++) {
		const g = verdictGrid(t, st, c, eng.h, eng.D, census);
		const cnt = [0, 0, 0];
		for (const x of g) cnt[x]++;
		const H = -cnt.reduce((a, k) => (k ? a + (k / g.length) * Math.log(k / g.length) : a), 0);
		if (H > bestH + 1e-9) {
			bestH = H;
			best = c;
		}
	}
	return best;
}

self.onmessage = async (ev: MessageEvent<Request>) => {
	const q = ev.data;
	if (q.type === 'load') {
		const [cfg, buf] = await Promise.all([
			fetch(`${q.base}/models/${q.name}/config.json`).then((r) => r.json() as Promise<ModelConfig>),
			fetch(`${q.base}/models/${q.name}/model.safetensors`).then((r) => r.arrayBuffer())
		]);
		engine?.destroy();
		engine = await makeRunner(parseSafetensors(buf), cfg, q.engine);
		post({ type: 'loaded', id: q.id, config: cfg, engine: engine.kind });
	} else if (q.type === 'stop') {
		stopAt = q.id;
	} else if (q.type === 'run') {
		if (!engine) return;
		const id = q.id;
		n = q.inputs.n;
		traces = [null];
		const eng = engine;
		let st = await eng.start(q.inputs);
		states = [st];
		const read = async (s: State) => Object.fromEntries(Object.entries(await eng.read(s)).map(([k, v]) => [k, v.pred]));
		post({ type: 'loop', id, loop: 0, changed: st.changed, pred: await read(st), ms: 0, informative: mostTelling(st, null, eng) } satisfies LoopMsg);
		const cap = eng.block === 'census' ? censusLoops(eng.model, n) : eng.block === 'pair_numbers' ? 4 * n : n;
		for (let l = 1; l <= cap; l++) {
			// let a newer request (stop, or a run that supersedes this one) in between loops
			await new Promise((r) => setTimeout(r, 0));
			if (stopAt >= id) return;
			const t0 = performance.now();
			st = await eng.step(st);
			const pred = await read(st);
			const ms = performance.now() - t0;
			if (stopAt >= id) return;
			states.push(st);
			traces.push(st.trace);
			post({ type: 'loop', id, loop: l, changed: st.changed, pred, ms, informative: mostTelling(st, st.trace, eng) } satisfies LoopMsg);
			if (eng.block !== 'census' && st.changed === 0) break;
		}
		post({ type: 'done', id, loops: st.loop, rested: eng.block !== 'census' && st.changed === 0, informative: mostTelling(st, traces[traces.length - 1], eng) });
	} else if (q.type === 'grid') {
		const t = traces[q.loop];
		const s = states[q.loop];
		const census = engine?.block === 'census';
		// the resting blocks' bits exist from loop 0; the general block's counts only after its first loop
		if (!s || !engine || (census && !t)) return post({ type: 'grid', id: q.id, grid: null });
		const g = verdictGrid(t, s, q.pair, engine.h, engine.D, census);
		post({ type: 'grid', id: q.id, grid: g }, [g.buffer]);
	} else if (q.type === 'pair') {
		const t = traces[q.loop];
		const s = states[q.loop];
		if (!t || !engine || !s) return post({ type: 'pair', id: q.id, info: null });
		const h = engine.h;
		const { u, v } = q;
		const witness = new Uint8Array(h * n);
		for (let c = 0; c < h; c++)
			for (let k = 0; k < n; k++) witness[c * n + k] = t.testA[(u * n + k) * h + c] && t.testB[(k * n + v) * h + c] ? 1 : 0;
		const i = u * n + v;
		const counts = t.counts.slice(i * h, (i + 1) * h);
		const best = t.best.slice(i * h, (i + 1) * h);
		const lead = t.lead ? t.lead.slice(i * (h / 2), (i + 1) * (h / 2)) : null;
		const D = engine.D;
		const bits = engine.block === 'census' ? new Float32Array(0) : s.s.slice(i * D, (i + 1) * D);
		const info: PairInfo = { witness, counts, lead, best, bits };
		post({ type: 'pair', id: q.id, info });
	}
};
