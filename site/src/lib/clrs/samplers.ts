/**
 * Fresh CLRS-30 instances, drawn as dm-clrs 2.0.3's samplers (clrs/_src/samplers.py) draw them: the same distributions
 * and default parameters, so a sampled instance looks like one of the official test or val instances. The random
 * stream is ours (rng.ts), not numpy's, so the numbers differ from the library's for the same seed.
 *
 * `length` is the library's sampler length. The instance's item count n (see itemsFor):
 *   - n = length for every task except
 *   - optimal_bst: n = length + 1 (length keys, length + 1 gaps),
 *   - segments_intersect: n = 4 always (two segments' endpoints; length is ignored),
 *   - the string matchers: n = length = haystack + needle; the needle has length // 5 characters (1 below length 5).
 *     The official splits use 5/4 of the nominal size here (80 for test's 64, 20 for val's 16): see splitLength.
 *   - lcs_length: n = length = |x| + |y| with |y| = length // 2.
 *
 * Positions are the library's deterministic ones (item i of n at i / n; for the two-string tasks each string from 0
 * separately). Training randomises them (randomizePos, the library's process_random_pos).
 */
import { isGraph, type GraphTask, type Instance, type Task, type TypedArray } from './instances';
import { solve } from './reference';
import type { Rng } from './rng';

export interface SampleOptions {
	/**
	 * Graph tasks: Erdos-Renyi edge probability, or a list to draw it from per instance (the library's `p` tuple).
	 * Defaults as the library's samplers: 0.2 for articulation_points, bridges and mst_kruskal, 0.5 otherwise.
	 */
	p?: number | readonly number[];
	/** String matchers: needle length; negative -k draws it uniformly from 1..k. Default length // 5 (1 below 5). */
	lengthNeedle?: number;
}

const HALF_DENSE: readonly Task[] = ['articulation_points', 'bridges', 'mst_kruskal'];
const MATCHERS: readonly Task[] = ['naive_string_matcher', 'kmp_matcher'];

/** Items of an instance drawn at sampler length `length`. */
export function itemsFor(task: Task, length: number): number {
	if (task === 'segments_intersect') return 4;
	if (task === 'optimal_bst') return length + 1;
	return length;
}

/** The sampler length the official splits use for a nominal size (64 test, 16 val): 5/4 of it for the matchers. */
export function splitLength(task: Task, size: number): number {
	return MATCHERS.includes(task) ? Math.floor((size * 5) / 4) : size;
}

/** The smallest sampler length the library's sampler accepts for a task. */
export function minLength(task: Task): number {
	if (task === 'strongly_connected_components') return 4; // four communities
	if (MATCHERS.includes(task)) return 3; // the haystack must be longer than the needle
	if (task === 'lcs_length') return 2;
	return 1;
}

// ---------------------------------------------------------------------------------------------------------------
// Graphs (row-major n*n float64)

interface ErOptions {
	directed: boolean;
	acyclic: boolean;
	weighted: boolean;
}

/** samplers._random_er_graph (low = 0, high = 1). */
function erGraph(rng: Rng, n: number, p: number, o: ErOptions): Float64Array {
	let mat: Float64Array = new Float64Array(n * n);
	for (let t = 0; t < n * n; t++) mat[t] = rng.bernoulli(p);
	if (!o.directed) {
		for (let i = 0; i < n; i++)
			for (let j = i; j < n; j++) mat[i * n + j] = mat[j * n + i] = mat[i * n + j] * mat[j * n + i];
	} else if (o.acyclic) {
		for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) mat[i * n + j] = 0;
		mat = permute(mat, n, rng.permutation(n));
	}
	if (o.weighted) {
		const w = new Float64Array(n * n);
		for (let t = 0; t < n * n; t++) w[t] = rng.uniform(0, 1);
		if (!o.directed) {
			for (let i = 0; i < n; i++)
				for (let j = i; j < n; j++) w[i * n + j] = w[j * n + i] = Math.sqrt(w[i * n + j] * w[j * n + i] + 1e-3);
		}
		for (let t = 0; t < n * n; t++) mat[t] *= w[t];
	}
	return mat;
}

/** mat[perm, :][:, perm]. */
function permute(mat: Float64Array, n: number, perm: Int32Array): Float64Array {
	const out = new Float64Array(n * n);
	for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[i * n + j] = mat[perm[i] * n + perm[j]];
	return out;
}

/** samplers._random_community_graph (directed, unweighted): k ER communities, sparse forward links, permuted. */
function communityGraph(rng: Rng, n: number, k: number, p: number, eps: number): Float64Array {
	if (k > n) throw new RangeError(`cannot make ${k} communities of ${n} nodes`);
	const mat = new Float64Array(n * n);
	const los: number[] = [];
	const his: number[] = [];
	let lo = 0;
	for (let c = 0; c < k; c++) {
		const hi = c === k - 1 ? n : lo + Math.floor(n / k);
		const m = hi - lo;
		const block = erGraph(rng, m, p, { directed: true, acyclic: false, weighted: false });
		for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) mat[(lo + i) * n + lo + j] = block[i * m + j];
		los.push(lo);
		his.push(hi);
		lo = hi;
	}
	const toggle = erGraph(rng, n, eps, { directed: true, acyclic: false, weighted: false });
	for (let c = 0; c < k; c++)
		for (let d = 0; d < c; d++)
			for (let i = los[c]; i < his[c]; i++) for (let j = los[d]; j < his[d]; j++) toggle[i * n + j] = 0;
	for (let t = 0; t < n * n; t++) if (toggle[t] > 0) mat[t] = (mat[t] > 0 ? 0 : 1) * toggle[t];
	return permute(mat, n, rng.permutation(n));
}

function graphInstance(task: GraphTask, n: number, mat: Float64Array, s?: number): Instance {
	const pos = new Float32Array(n);
	for (let i = 0; i < n; i++) pos[i] = (i * 1.0) / n;
	const A = Float32Array.from(mat);
	const adj = new Uint8Array(n * n);
	for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) adj[i * n + j] = A[i * n + j] + (i === j ? 1 : 0) !== 0 ? 1 : 0;
	const inputs: Instance['inputs'] = { pos, A, adj };
	if (s !== undefined) inputs.s = s;
	return { task, n, inputs, outputs: {} };
}

function sampleGraph(task: GraphTask, n: number, rng: Rng, opts: SampleOptions): Instance {
	const ps = opts.p ?? (HALF_DENSE.includes(task) ? 0.2 : 0.5);
	const p = typeof ps === 'number' ? ps : rng.choice(ps);
	const er = (o: ErOptions) => erGraph(rng, n, p, o);
	switch (task) {
		case 'bfs':
			return graphInstance(task, n, er({ directed: false, acyclic: false, weighted: false }), rng.int(n));
		case 'dfs':
			return graphInstance(task, n, er({ directed: true, acyclic: false, weighted: false }));
		case 'topological_sort':
			return graphInstance(task, n, er({ directed: true, acyclic: true, weighted: false }));
		case 'articulation_points':
		case 'bridges':
			return graphInstance(task, n, er({ directed: false, acyclic: false, weighted: false }));
		case 'mst_kruskal':
		case 'floyd_warshall':
			return graphInstance(task, n, er({ directed: false, acyclic: false, weighted: true }));
		case 'bellman_ford':
		case 'dijkstra':
		case 'mst_prim':
			return graphInstance(task, n, er({ directed: false, acyclic: false, weighted: true }), rng.int(n));
		case 'dag_shortest_paths':
			return graphInstance(task, n, er({ directed: true, acyclic: true, weighted: true }), rng.int(n));
		case 'strongly_connected_components':
			return graphInstance(task, n, communityGraph(rng, n, 4, p, 0.01));
	}
}

// ---------------------------------------------------------------------------------------------------------------
// Arrays, strings, points

const positions = (n: number): Float64Array => Float64Array.from({ length: n }, (_, i) => (i * 1.0) / n);
const uniforms = (rng: Rng, n: number, low = 0, high = 1): Float64Array =>
	Float64Array.from({ length: n }, () => rng.uniform(low, high));
const randomString = (rng: Rng, n: number, chars = 4): Int32Array =>
	Int32Array.from({ length: n }, () => rng.int(chars));

/** Two strings in one sequence as the library lays them out (probing.strings_id / strings_pos / array_cat). */
function twoStringInputs(a: Int32Array, b: Int32Array): Instance['inputs'] {
	const n = a.length + b.length;
	const string = new Uint8Array(n);
	const pos = new Float64Array(n);
	const key = new Uint8Array(n * 4);
	for (let i = 0; i < a.length; i++) {
		pos[i] = (i * 1.0) / a.length;
		key[i * 4 + a[i]] = 1;
	}
	for (let j = 0; j < b.length; j++) {
		string[a.length + j] = 1;
		pos[a.length + j] = (j * 1.0) / b.length;
		key[(a.length + j) * 4 + b[j]] = 1;
	}
	return { string, pos, key };
}

/** SegmentsSampler's quick test (ignores collinearity) used to decide whether a draw intersects. */
function quickIntersect(xs: Float64Array, ys: Float64Array): number {
	const ccw = (a: number, b: number, c: number) =>
		(ys[c] - ys[a]) * (xs[b] - xs[a]) > (ys[b] - ys[a]) * (xs[c] - xs[a]);
	return ccw(0, 2, 3) !== ccw(1, 2, 3) && ccw(0, 1, 2) !== ccw(0, 1, 3) ? 1 : 0;
}

/** ConvexHullSampler: points uniform over the disk of radius 2, redrawn while any three are collinear (|cross| < 1e-12). */
function hullPoints(rng: Rng, n: number): [Float64Array, Float64Array] {
	for (let attempt = 0; attempt < 1000; attempt++) {
		const thetas = uniforms(rng, n, 0, 2 * Math.PI);
		const rs = uniforms(rng, n).map((u) => 2.0 * Math.sqrt(u));
		const xs = rs.map((r, i) => r * Math.cos(thetas[i]) + 0.0);
		const ys = rs.map((r, i) => r * Math.sin(thetas[i]) + 0.0);
		let collinear = false;
		search: for (let i = 0; i < n; i++)
			for (let j = i + 1; j < n; j++)
				for (let k = j + 1; k < n; k++) {
					const v1x = xs[j] - xs[i];
					const v1y = ys[j] - ys[i];
					const v2x = xs[k] - xs[i];
					const v2y = ys[k] - ys[i];
					if (Math.abs(v1x * v2y - v1y * v2x) < 1e-12) {
						collinear = true;
						break search;
					}
				}
		if (!collinear) return [xs, ys];
	}
	throw new Error(`could not sample ${n} points without collinear triples`);
}

function sampleSeq(task: Exclude<Task, GraphTask>, L: number, rng: Rng, opts: SampleOptions): Instance {
	const inst = (n: number, inputs: Instance['inputs']): Instance => ({ task, n, inputs, outputs: {} });
	switch (task) {
		case 'insertion_sort':
		case 'bubble_sort':
		case 'heapsort':
		case 'quicksort':
		case 'minimum':
		case 'quickselect':
			return inst(L, { pos: positions(L), key: uniforms(rng, L) });
		case 'matrix_chain_order':
			return inst(L, { pos: positions(L), p: uniforms(rng, L) });
		case 'binary_search': {
			const key = uniforms(rng, L).sort();
			return inst(L, { pos: positions(L), key, target: rng.uniform(0, 1) });
		}
		case 'find_maximum_subarray_kadane':
			return inst(L, { pos: positions(L), key: uniforms(rng, L, -1, 1) });
		case 'optimal_bst': {
			const arr = uniforms(rng, 2 * L + 1);
			let sum = 0;
			for (const v of arr) sum += v;
			for (let i = 0; i < arr.length; i++) arr[i] /= sum;
			const p = new Float64Array(L + 1);
			p.set(arr.subarray(0, L));
			return inst(L + 1, { pos: positions(L + 1), p, q: arr.slice(L) });
		}
		case 'lcs_length': {
			const lb = Math.floor(L / 2);
			const a = randomString(rng, L - lb);
			const b = randomString(rng, lb);
			return inst(L, twoStringInputs(a, b));
		}
		case 'activity_selector': {
			const u = uniforms(rng, L);
			const v = uniforms(rng, L);
			return inst(L, { pos: positions(L), s: u.map((x, i) => Math.min(x, v[i])), f: u.map((x, i) => Math.max(x, v[i])) });
		}
		case 'task_scheduling':
			return inst(L, {
				pos: positions(L),
				d: Float64Array.from(randomString(rng, L, L), (x) => x + 1),
				w: uniforms(rng, L)
			});
		case 'naive_string_matcher':
		case 'kmp_matcher': {
			let ln = opts.lengthNeedle ?? (L < 5 ? 1 : Math.floor(L / 5));
			if (ln < 0) ln = 1 + rng.int(-ln);
			const lh = L - ln;
			if (!(ln >= 1 && lh > ln)) throw new RangeError(`needle ${ln} in a haystack of ${lh}`);
			const needle = randomString(rng, ln);
			const hay = randomString(rng, lh);
			hay.set(needle, rng.int(lh - ln));
			return inst(L, twoStringInputs(hay, needle));
		}
		case 'segments_intersect': {
			const coin = rng.bernoulli(0.5);
			let xs = uniforms(rng, 4);
			let ys = uniforms(rng, 4);
			while (quickIntersect(xs, ys) !== coin) {
				xs = uniforms(rng, 4);
				ys = uniforms(rng, 4);
			}
			return inst(4, { pos: positions(4), x: xs, y: ys });
		}
		case 'graham_scan':
		case 'jarvis_march': {
			const [x, y] = hullPoints(rng, L);
			return inst(L, { pos: positions(L), x, y });
		}
	}
}

/** Inputs of a fresh instance (outputs left empty; see sample). */
export function sampleInputs(task: Task, length: number, rng: Rng, opts: SampleOptions = {}): Instance {
	if (!Number.isInteger(length) || length < minLength(task))
		throw new RangeError(`${task} needs an integer length >= ${minLength(task)}, got ${length}`);
	return isGraph(task) ? sampleGraph(task, length, rng, opts) : sampleSeq(task, length, rng, opts);
}

/** A fresh instance with its true outputs (reference.solve). */
export function sample(task: Task, length: number, rng: Rng, opts: SampleOptions = {}): Instance {
	const inst = sampleInputs(task, length, rng, opts);
	inst.outputs = solve(task, inst);
	return inst;
}

/**
 * The library's process_random_pos (training only): `pos` replaced by sorted uniforms within each run of items that
 * starts where pos is 0 (the whole instance, or each of the two strings). Returns a copy; pos keeps its dtype.
 */
export function randomizePos(inst: Instance, rng: Rng): Instance {
	const pos = inst.inputs.pos as TypedArray;
	const out = pos.slice();
	const starts: number[] = [];
	for (let i = 0; i < pos.length; i++) if (pos[i] === 0) starts.push(i);
	starts.push(pos.length);
	for (let t = 0; t + 1 < starts.length; t++) {
		const run = Array.from({ length: starts[t + 1] - starts[t] }, () => rng.random()).sort((a, b) => a - b);
		out.set(run, starts[t]);
	}
	return { ...inst, inputs: { ...inst.inputs, pos: out } };
}
