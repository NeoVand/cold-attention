import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, test } from 'vitest';
import {
	decodeArray,
	decodeSplit,
	encodeArray,
	instanceAt,
	instances,
	isGraph,
	schema,
	TASKS,
	toPadded,
	type Instance,
	type Split,
	type SplitJson,
	type Task,
	type TypedArray,
	type Value
} from './instances';
import { diffOutputs, solve } from './reference';
import { itemsFor, randomizePos, sample, sampleInputs, splitLength } from './samplers';
import { makeRng } from './rng';

const ROOT = fileURLToPath(new URL('../../../static/instances/', import.meta.url));
const SPLITS = ['test', 'val'] as const;
const cache = new Map<string, Split>();
function load(task: Task, split: string): Split {
	const key = `${task}/${split}`;
	let s = cache.get(key);
	if (!s) {
		s = decodeSplit(JSON.parse(readFileSync(`${ROOT}${key}.json`, 'utf8')) as SplitJson);
		cache.set(key, s);
	}
	return s;
}

const arr = (v: Value | undefined): TypedArray => {
	if (v === undefined || typeof v === 'number') throw new Error('not an array');
	return v;
};
const mean = (a: ArrayLike<number>): number => {
	let t = 0;
	for (let i = 0; i < a.length; i++) t += a[i];
	return t / a.length;
};
const MATCHERS: readonly Task[] = ['naive_string_matcher', 'kmp_matcher'];
const STRING_TASKS: readonly Task[] = [...MATCHERS, 'lcs_length'];

/** How the library lays out an instance's inputs (positions, adjacency, string split, sampler invariants). */
function layoutProblems(inst: Instance): string[] {
	const { task, n } = inst;
	const x = inst.inputs;
	const bad: string[] = [];
	const sch = schema(task);
	for (const f of sch.inputs) {
		const v = x[f.name];
		if (v === undefined) bad.push(`missing ${f.name}`);
		else if (f.kind === 'graph') {
			if (typeof v !== 'number') bad.push(`${f.name} not a number`);
		} else {
			const len = f.kind === 'node' ? n : f.kind === 'edge' ? n * n : n * (f.classes ?? 0);
			const ctor = { f32: Float32Array, f64: Float64Array, i16: Int16Array, u8: Uint8Array }[f.dtype];
			if (!(v instanceof ctor) || v.length !== len) bad.push(`${f.name}: wrong dtype or length`);
		}
	}
	if (bad.length) return bad;
	const pos = arr(x.pos);
	if (isGraph(task)) {
		const A = arr(x.A);
		const adj = arr(x.adj);
		for (let i = 0; i < n; i++) {
			if (pos[i] !== Math.fround(i / n)) bad.push(`pos[${i}]`);
			for (let j = 0; j < n; j++) {
				if (adj[i * n + j] !== (A[i * n + j] + (i === j ? 1 : 0) !== 0 ? 1 : 0)) bad.push(`adj[${i},${j}]`);
				if (A[i * n + j] < 0) bad.push(`A[${i},${j}] < 0`);
				const undirected = !['dfs', 'topological_sort', 'dag_shortest_paths', 'strongly_connected_components'].includes(task);
				if (undirected && A[i * n + j] !== A[j * n + i]) bad.push(`A not symmetric at ${i},${j}`);
			}
		}
		if (typeof x.s === 'number' && !(x.s >= 0 && x.s < n)) bad.push('source out of range');
		if (task === 'topological_sort' || task === 'dag_shortest_paths') {
			// acyclic: Kahn's algorithm removes every node
			const indeg = new Int32Array(n);
			for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (A[i * n + j] !== 0) indeg[j]++;
			const queue = [...indeg.keys()].filter((j) => indeg[j] === 0);
			let seen = 0;
			while (queue.length) {
				const u = queue.pop()!;
				seen++;
				for (let v = 0; v < n; v++) if (A[u * n + v] !== 0 && --indeg[v] === 0) queue.push(v);
			}
			if (seen !== n) bad.push('graph has a cycle');
		}
		return bad;
	}
	if (STRING_TASKS.includes(task)) {
		const str = arr(x.string);
		const key = arr(x.key);
		const na = str.findIndex((v) => v === 1);
		if (na <= 0) bad.push('no second string');
		const nb = n - na;
		for (let i = 0; i < n; i++) {
			if (str[i] !== (i < na ? 0 : 1)) bad.push(`string[${i}]`);
			const want = i < na ? i / na : (i - na) / nb;
			if (pos[i] !== want) bad.push(`pos[${i}]`);
			let ones = 0;
			for (let c = 0; c < 4; c++) ones += key[i * 4 + c];
			if (ones !== 1) bad.push(`key[${i}] not one-hot`);
		}
		if (task === 'lcs_length' && nb !== Math.floor(n / 2)) bad.push(`|y| = ${nb}`);
		if (MATCHERS.includes(task) && nb !== (n < 5 ? 1 : Math.floor(n / 5))) bad.push(`needle of ${nb}`);
		return bad;
	}
	for (let i = 0; i < n; i++) if (pos[i] !== i / n) bad.push(`pos[${i}]`);
	if (task === 'binary_search') {
		const key = arr(x.key);
		for (let i = 1; i < n; i++) if (key[i - 1] > key[i]) bad.push('key not sorted');
	}
	if (task === 'optimal_bst') {
		const p = arr(x.p);
		const q = arr(x.q);
		if (p[n - 1] !== 0) bad.push('p not padded with 0');
		const total = [...p, ...q].reduce((a, b) => a + b, 0);
		if (Math.abs(total - 1) > 1e-9) bad.push(`probabilities sum to ${total}`);
	}
	if (task === 'activity_selector') {
		const s = arr(x.s);
		const f = arr(x.f);
		for (let i = 0; i < n; i++) if (!(s[i] <= f[i])) bad.push(`activity ${i} ends before it starts`);
	}
	if (task === 'task_scheduling') {
		const d = arr(x.d);
		for (let i = 0; i < n; i++) if (!(Number.isInteger(d[i]) && d[i] >= 1 && d[i] <= n)) bad.push(`deadline ${d[i]}`);
	}
	if (task === 'graham_scan' || task === 'jarvis_march') {
		const xs = arr(x.x);
		const ys = arr(x.y);
		for (let i = 0; i < n; i++) if (Math.hypot(xs[i], ys[i]) > 2 + 1e-12) bad.push(`point ${i} outside the disk`);
	}
	if (task === 'find_maximum_subarray_kadane') {
		const key = arr(x.key);
		for (let i = 0; i < n; i++) if (!(key[i] >= -1 && key[i] < 1)) bad.push(`key[${i}]`);
	}
	return bad;
}

/** Output shapes and ranges, and a few semantic checks of the answer. */
function outputProblems(inst: Instance): string[] {
	const { task, n } = inst;
	const bad: string[] = [];
	const out = inst.outputs;
	for (const f of schema(task).outputs) {
		const v = out[f.name];
		if (f.kind === 'graph') {
			if (typeof v !== 'number') bad.push(`${f.name} missing`);
			continue;
		}
		const a = arr(v);
		if (!(a instanceof Int16Array) || a.length !== (f.kind === 'edge' ? n * n : n)) bad.push(`${f.name} shape`);
	}
	if (bad.length) return bad;
	const inRange = (v: number) => Number.isInteger(v) && v >= 0 && v < n;
	const x = inst.inputs;
	switch (task) {
		case 'insertion_sort':
		case 'bubble_sort':
		case 'heapsort':
		case 'quicksort': {
			const pred = arr(out.pred);
			const key = arr(x.key);
			const heads = [...pred.keys()].filter((i) => pred[i] === i);
			if (heads.length !== 1) bad.push('pred has no single head');
			// follow successors from the head: every key, ascending
			const succ = new Map<number, number>();
			pred.forEach((p, i) => p !== i && succ.set(p, i));
			let cur = heads[0];
			let count = 1;
			while (succ.has(cur)) {
				const nx = succ.get(cur)!;
				if (key[nx] < key[cur]) bad.push('pred order not sorted');
				cur = nx;
				count++;
			}
			if (count !== n) bad.push('pred is not one chain');
			break;
		}
		case 'minimum': {
			const key = arr(x.key);
			if (key.some((k) => k < key[out.min as number])) bad.push('min is not the minimum');
			break;
		}
		case 'quickselect': {
			const key = [...arr(x.key)].sort((a, b) => a - b);
			if (arr(x.key)[out.median as number] !== key[Math.floor(n / 2)]) bad.push('median is not rank n // 2');
			break;
		}
		case 'binary_search': {
			const key = arr(x.key);
			const r = out.return as number;
			const t = x.target as number;
			if (!inRange(r) || (r < n - 1 && key[r] < t) || (r > 0 && key[r - 1] >= t)) bad.push('return is not the insertion point');
			break;
		}
		case 'naive_string_matcher':
		case 'kmp_matcher': {
			const str = arr(x.string);
			const key = arr(x.key);
			const ch = (i: number) => [0, 1, 2, 3].find((c) => key[i * 4 + c] === 1);
			const nh = str.indexOf(1);
			const m = out.match as number;
			for (let j = 0; j < n - nh; j++) if (ch(m + j) !== ch(nh + j)) bad.push('match is not a match');
			break;
		}
		case 'topological_sort': {
			// following topo from topo_head visits every node, edges point forward
			const topo = arr(out.topo);
			const A = arr(x.A);
			const rank = new Int32Array(n).fill(-1);
			let cur = out.topo_head as number;
			for (let r = 0; r < n; r++) {
				rank[cur] = r;
				if (topo[cur] === cur) break;
				cur = topo[cur];
			}
			if (rank.some((r) => r < 0)) bad.push('topo misses nodes');
			for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (A[i * n + j] && rank[i] > rank[j]) bad.push('edge backwards');
			break;
		}
		default:
			for (const f of schema(task).outputs) {
				const v = out[f.name];
				const vals = typeof v === 'number' ? [v] : [...arr(v)];
				const ok =
					f.name === 'is_bridge' || f.name === 'b'
						? (u: number) => u >= -1 && u <= (f.name === 'b' ? 2 : 1)
						: ['is_cut', 'in_mst', 'selected', 'in_hull', 'intersect'].includes(f.name)
							? (u: number) => u === 0 || u === 1
							: inRange;
				if (!vals.every(ok)) bad.push(`${f.name} out of range`);
			}
	}
	return bad;
}

describe.each(TASKS)('%s', (task) => {
	test.each(SPLITS)("solve reproduces the library's outputs on %s", (split) => {
		const sp = load(task, split);
		const mismatches: string[] = [];
		for (let i = 0; i < sp.count; i++) {
			const inst = instanceAt(sp, i);
			const d = diffOutputs(solve(task, inst), inst.outputs);
			if (d.length) mismatches.push(`#${i}: ${d.slice(0, 4).join('; ')}${d.length > 4 ? ` (+${d.length - 4})` : ''}`);
		}
		expect(sp.count).toBeGreaterThan(0);
		expect(mismatches).toEqual([]);
	});

	test.each(SPLITS)('%s inputs are laid out as the samplers lay them out; toPadded restores the file', (split) => {
		const sp = load(task, split);
		const list = instances(sp);
		const problems = list.flatMap((inst, i) => layoutProblems(inst).map((p) => `#${i} ${p}`));
		expect(problems.slice(0, 10)).toEqual([]);
		const packed = toPadded(list, sp.M);
		expect(Object.keys(packed)).toEqual(Object.keys(sp.arrays));
		for (const [name, a] of Object.entries(sp.arrays)) {
			expect(packed[name].dtype, name).toBe(a.dtype);
			expect(packed[name].shape, name).toEqual(a.shape);
			expect(encodeArray(packed[name]).b64 === encodeArray(a).b64, name).toBe(true);
		}
	});

	test('sample: valid instances at sizes 4, 8, 16, 32 that solve runs on', () => {
		const rng = makeRng(TASKS.indexOf(task) + 1);
		for (const size of [4, 8, 16, 32]) {
			const L = splitLength(task, size);
			for (let r = 0; r < 12; r++) {
				const inst = sample(task, L, rng);
				expect(inst.n).toBe(itemsFor(task, L));
				const problems = [...layoutProblems(inst), ...outputProblems(inst)];
				expect(problems, `size ${size} draw ${r}`).toEqual([]);
				// padded round trip
				const packed = toPadded([inst], inst.n + 3);
				const back = instanceAt({ task, split: 'sample', count: 1, M: inst.n + 3, arrays: packed }, 0);
				expect(back).toEqual(inst);
			}
		}
		const again = (seed: number) => sample(task, splitLength(task, 8), makeRng(seed));
		expect(again(7)).toEqual(again(7));
	});

	test('sample: same distribution as the val split (per-instance means within 5 standard errors)', () => {
		const sp = load(task, 'val');
		const official = instances(sp);
		const L = splitLength(task, 16);
		expect(official[0].n).toBe(itemsFor(task, L));
		const rng = makeRng(1000 + TASKS.indexOf(task));
		const drawn = Array.from({ length: 400 }, () => sample(task, L, rng));
		const fields: ['inputs' | 'outputs', string][] = [
			...schema(task).inputs.map((f): ['inputs', string] => ['inputs', f.name]),
			...schema(task).outputs.map((f): ['outputs', string] => ['outputs', f.name])
		];
		const far: string[] = [];
		for (const [where, name] of fields) {
			const stat = (list: Instance[]) => {
				const means = list.map((inst) => {
					const v = inst[where][name];
					return typeof v === 'number' ? v : mean(arr(v));
				});
				const m = means.reduce((a, b) => a + b, 0) / means.length;
				const v = means.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, means.length - 1);
				return { m, se2: v / means.length };
			};
			const a = stat(official);
			const b = stat(drawn);
			const se = Math.sqrt(a.se2 + b.se2);
			if (Math.abs(a.m - b.m) > 5 * se + 1e-12) far.push(`${name}: val ${a.m.toFixed(4)} vs sampled ${b.m.toFixed(4)} (se ${se.toFixed(4)})`);
		}
		expect(far).toEqual([]);
	});
});

test("sample takes run.py's training options (p drawn from a list, random needle lengths)", () => {
	const rng = makeRng(11);
	for (let r = 0; r < 50; r++) {
		const inst = sample('kmp_matcher', 20, rng, { lengthNeedle: -8 });
		const needle = Array.from(arr(inst.inputs.string)).filter((v) => v === 1).length;
		expect(needle >= 1 && needle <= 8).toBe(true);
		expect(outputProblems(inst)).toEqual([]);
		const g = sample('bridges', 16, rng, { p: [0.05, 0.45] });
		expect([...layoutProblems(g), ...outputProblems(g)]).toEqual([]);
	}
	expect(() => sample('kmp_matcher', 2, rng)).toThrow(RangeError);
});

test('randomizePos sorts uniforms within each string', () => {
	const inst = sample('kmp_matcher', 20, makeRng(3));
	const r = randomizePos(inst, makeRng(4));
	const pos = arr(r.inputs.pos);
	const str = arr(r.inputs.string);
	for (let i = 1; i < inst.n; i++) if (str[i] === str[i - 1]) expect(pos[i]).toBeGreaterThanOrEqual(pos[i - 1]);
	expect(r.outputs).toEqual(inst.outputs);
	expect(pos).toBeInstanceOf(Float64Array);
	expect(arr(randomizePos(sampleInputs('bfs', 8, makeRng(5)), makeRng(6)).inputs.pos)).toBeInstanceOf(Float32Array);
});

test('encodeArray and decodeArray are inverse', () => {
	const a = { dtype: 'f64' as const, shape: [2, 3], data: Float64Array.from([0.1, -2, 3e-300, 1 / 3, -0, 7]) };
	const back = decodeArray(encodeArray(a));
	expect(back).toEqual(a);
	const i16 = (shape: number[], v: number[]) =>
		encodeArray({ dtype: 'i16', shape, data: Int16Array.from(v) });
	const f64 = (shape: number[], v: number[]) =>
		encodeArray({ dtype: 'f64', shape, data: Float64Array.from(v) });
	const json: SplitJson = {
		task: 'minimum',
		split: 'x',
		count: 1,
		arrays: { n: i16([1], [2]), x_pos: f64([1, 2], [0, 0.5]), x_key: f64([1, 2], [0.3, 0.2]), min: i16([1], [1]) }
	};
	const sp = decodeSplit(json);
	expect(sp.M).toBe(2);
	expect(solve('minimum', instanceAt(sp, 0))).toEqual({ min: 1 });
});
