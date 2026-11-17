/**
 * The page's state: which model, which instance, the loops as they arrive from the worker, and what the reader has
 * selected (a pair, a channel pair, a loop).
 */
import type { ModelConfig } from '../model/engine';
import { decodeArrays, instanceArrays, modelInputs, type Arrays, type SplitJSON } from '../model/inputs';
import type { LoopMsg, PairInfo, Request } from './worker';
import { TASK, PREFERRED_BLOCK } from './tasks';

export interface ModelEntry {
	name: string;
	task: string;
	block: string;
	scores: Record<string, number>;
	parameters: number;
}

export interface Instance {
	/** the task it is an instance of (the view follows it, not the task picked, so they never disagree) */
	task: string;
	n: number;
	arrays: Arrays;
	/** where it came from, in words */
	origin: string;
}

type DistributiveOmit<T, K extends keyof T> = T extends unknown ? Omit<T, K> : never;
type Pending = (m: { type: string; [k: string]: unknown }) => void;

export class Demo {
	base: string;
	index = $state.raw<ModelEntry[]>([]);
	/** every released model's scores (all seeds), for the score grid */
	scores = $state.raw<{ task: string; block: string; seed: number; test: number }[]>([]);
	task = $state('bellman_ford');
	block = $state('census');
	seed = $state(0);
	source = $state<'official' | 'fresh'>('official');
	split = $state<'test' | 'val'>('test');
	which = $state(0);
	size = $state(16);
	instance = $state.raw<Instance | null>(null);
	config = $state.raw<ModelConfig | null>(null);
	loops = $state.raw<LoopMsg[]>([]);
	shown = $state(0);
	running = $state(false);
	rested = $state<boolean | null>(null);
	loading = $state('');
	sel = $state.raw<{ u: number; v: number } | null>(null);
	/** the first item of a pair being picked (the second click makes the pair) */
	pending = $state<number | null>(null);
	chPair = $state(0);
	grid = $state.raw<Uint8Array | null>(null);
	pair = $state.raw<PairInfo | null>(null);
	splitCount = $state(0);
	/** the engine asked for, and the one running (WebGPU when the browser has it) */
	engine = $state<'gpu' | 'cpu'>('gpu');
	engineUsed = $state('');

	private worker: Worker | null = null;
	/** loops are revealed at a steady pace (at most one every PACE ms), so a fast engine still animates */
	private queue: { type: string; [k: string]: unknown }[] = [];
	private pumping: ReturnType<typeof setTimeout> | null = null;
	private lastShown = 0;
	static PACE = 160;

	private pump() {
		if (this.pumping) return;
		const next = () => {
			this.pumping = null;
			const m = this.queue.shift();
			if (!m) return;
			if (m.type === 'loop') {
				const l = m as unknown as LoopMsg;
				const follow = this.shown === (this.loops.at(-1)?.loop ?? 0);
				this.loops = [...this.loops, l];
				if (follow) {
					this.shown = l.loop;
					if (!this.chosenByReader) this.chPair = l.informative;
					this.refreshDetail();
				}
				this.lastShown = performance.now();
			} else {
				this.running = false;
				this.rested = (m.rested as boolean) ?? null;
				if (!this.chosenByReader) this.chPair = (m.informative as number) ?? 0;
				this.refreshDetail();
			}
			if (this.queue.length) this.pumping = setTimeout(next, Demo.PACE);
		};
		const wait = Math.max(0, Demo.PACE - (performance.now() - this.lastShown));
		this.pumping = setTimeout(next, wait);
	}
	private nextId = 1;
	private runId = 0;
	private waiting = new Map<number, Pending>();
	private splits = new Map<string, Arrays>();

	constructor(base: string) {
		this.base = base;
	}

	/** the blocks this task has models for: the preferred one first, then the census block */
	blocks = $derived.by(() => {
		const have = new Set(this.index.filter((m) => m.task === this.task).map((m) => m.block));
		const first = PREFERRED_BLOCK[this.task] ?? 'census';
		return [first, 'census', 'shared', 'cold', 'compare_first', 'pair_numbers'].filter((b, i, a) => have.has(b) && a.indexOf(b) === i);
	});
	seeds = $derived(
		this.index
			.filter((m) => m.task === this.task && m.block === this.block)
			.map((m) => Number(m.name.split('seed')[1]))
			.sort()
	);
	model = $derived(this.index.find((m) => m.name === `${this.block}/${this.task}/seed${this.seed}`) ?? null);
	current = $derived(this.loops.find((l) => l.loop === this.shown) ?? this.loops.at(-1) ?? null);
	info = $derived(TASK[this.task]);

	async init() {
		this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
		this.worker.onmessage = (ev) => this.onMessage(ev.data);
		[this.index, this.scores] = await Promise.all([
			fetch(`${this.base}/models/index.json`).then((r) => r.json()),
			fetch(`${this.base}/models/scores.json`).then((r) => r.json())
		]);
		await this.choose(this.task);
	}

	destroy() {
		this.worker?.terminate();
		this.worker = null;
	}

	private onMessage(m: { type: string; id: number; [k: string]: unknown }) {
		if (m.type === 'loop' || m.type === 'done') {
			if (m.id !== this.runId) return;
			this.queue.push(m);
			this.pump();
			return;
		}
		const cb = this.waiting.get(m.id);
		if (cb) {
			this.waiting.delete(m.id);
			cb(m);
		}
	}

	private ask(q: DistributiveOmit<Request, 'id'>): Promise<{ type: string; [k: string]: unknown }> {
		const id = this.nextId++;
		return new Promise((res) => {
			this.waiting.set(id, res);
			this.worker!.postMessage({ ...q, id } as Request);
		});
	}

	/** Pick a task: its first block and seed, an official instance. */
	async choose(task: string) {
		this.task = task;
		this.chosenByReader = false;
		this.block = this.blocks[0] ?? 'census';
		if (!this.seeds.includes(this.seed)) this.seed = this.seeds[0] ?? 0;
		const [, lo, hi] = TASK[task].size;
		this.size = Math.min(hi, Math.max(lo, this.size));
		await this.loadModel();
		await this.loadInstance();
	}

	async setBlock(block: string) {
		this.block = block;
		if (!this.seeds.includes(this.seed)) this.seed = this.seeds[0] ?? 0;
		await this.loadModel();
		this.run();
	}

	async setEngine(engine: 'gpu' | 'cpu') {
		this.engine = engine;
		await this.loadModel();
		this.run();
	}

	async setSeed(seed: number) {
		this.seed = seed;
		await this.loadModel();
		this.run();
	}

	async loadModel() {
		if (!this.model) return;
		this.loading = 'model';
		const m = await this.ask({ type: 'load', base: this.base, name: this.model.name, engine: this.engine });
		this.config = m.config as ModelConfig;
		this.engineUsed = m.engine as string;
		this.loading = '';
	}

	/** An official instance (index `which` of the split), or a fresh one from `fresh`. */
	async loadInstance(fresh?: () => Instance) {
		if (fresh) {
			this.instance = fresh();
		} else {
			const key = `${this.task}/${this.split}`;
			if (!this.splits.has(key)) {
				this.loading = 'instances';
				const doc = (await fetch(`${this.base}/instances/${key}.json`).then((r) => r.json())) as SplitJSON;
				this.splits.set(key, decodeArrays(doc));
				this.loading = '';
			}
			const d = this.splits.get(key)!;
			this.splitCount = d.n.data.length;
			this.which = Math.min(this.which, this.splitCount - 1);
			const { n, arrays } = instanceArrays(d, this.which);
			this.instance = { task: this.task, n, arrays, origin: `official ${this.split} instance ${this.which + 1} of ${this.splitCount}` };
		}
		this.sel = null;
		this.pending = null;
		this.run();
	}

	run() {
		if (!this.instance || !this.config || !this.worker || this.config.task !== this.instance.task) return;
		this.worker.postMessage({ type: 'stop', id: this.runId } as Request);
		this.runId = this.nextId++;
		this.queue = [];
		if (this.pumping) clearTimeout(this.pumping);
		this.pumping = null;
		this.loops = [];
		this.shown = 0;
		this.running = true;
		this.rested = null;
		this.grid = null;
		this.pair = null;
		const inputs = modelInputs(this.config.model, this.instance.n, this.instance.arrays);
		this.worker.postMessage({ type: 'run', id: this.runId, inputs } as Request);
	}

	show(loop: number) {
		this.shown = loop;
		this.refreshDetail();
	}

	/** select the pair (u, v); unless the reader chose a channel pair, show one with a few learned witnesses for it */
	select(u: number, v: number) {
		this.sel = { u, v };
		this.pending = null;
		if (!this.chosenByReader) this.sparseChannel = true;
		this.refreshDetail();
	}

	/** a click on an item: the first makes it u, the second (another item) makes the pair; the same item again cancels */
	pickItem(k: number) {
		if (this.pending === null) {
			this.sel = null;
			this.pair = null;
			this.pending = k;
		} else if (this.pending === k) this.pending = null;
		else this.select(this.pending, k);
	}

	clearSel() {
		this.sel = null;
		this.pending = null;
		this.pair = null;
	}

	private chosenByReader = false;
	/** after a pick: show the channel pair with the fewest witnesses (1 to 5) for that pair, once */
	private sparseChannel = false;

	setChPair(c: number) {
		this.chosenByReader = true;
		this.chPair = c;
		this.refreshDetail();
	}

	/** The verdict grid of the selected channel pair, and the selected pair's witnesses, at the loop shown. */
	async refreshDetail() {
		if (!this.worker || !this.loops.length) {
			this.grid = null;
			this.pair = null;
			return;
		}
		const loop = this.shown;
		const g = await this.ask({ type: 'grid', loop, pair: this.chPair });
		if (loop === this.shown) this.grid = (g.grid as Uint8Array) ?? null;
		if (this.sel) {
			const p = await this.ask({ type: 'pair', loop, u: this.sel.u, v: this.sel.v });
			const info = (p.info as PairInfo) ?? null;
			if (info && this.sparseChannel) {
				this.sparseChannel = false;
				let bestC = -1;
				let bestW = Infinity;
				const n = this.instance!.n;
				const { u, v } = this.sel;
				for (let c = 0; c < info.counts.length / 2; c++) {
					const a = info.counts[2 * c];
					const w = a + info.counts[2 * c + 1];
					// a learned channel: some witness other than the pair's own two items
					let middle = false;
					for (let k = 0; k < n; k++) if (k !== u && k !== v && info.witness[2 * c * n + k]) middle = true;
					if (middle && a >= 1 && a <= 5 && w < bestW) {
						bestW = w;
						bestC = c;
					}
				}
				if (bestC >= 0 && bestC !== this.chPair) {
					this.chPair = bestC;
					const g2 = await this.ask({ type: 'grid', loop, pair: bestC });
					if (loop === this.shown) this.grid = (g2.grid as Uint8Array) ?? null;
				}
			}
			if (loop === this.shown) this.pair = info;
		} else this.pair = null;
	}
}
