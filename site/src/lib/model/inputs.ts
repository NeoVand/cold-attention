/**
 * Model inputs for one instance, from split arrays in the layout of the official files (padded numpy arrays, see
 * scripts/export_instances.py). A port of witness_clrs/data.py's model_inputs for one instance, without padding.
 */
import { BIG, type Inputs, type ModelConfig } from './engine';

export interface NDArray {
	dtype: 'f32' | 'f64' | 'i16' | 'u8';
	shape: number[];
	data: Float32Array | Float64Array | Int16Array | Uint8Array;
}
export type Arrays = Record<string, NDArray>;

export interface SplitJSON {
	task: string;
	split: string;
	count: number;
	arrays: Record<string, { dtype: NDArray['dtype']; shape: number[]; b64: string }>;
}

function b64bytes(s: string): Uint8Array {
	if (typeof atob === 'function') {
		const bin = atob(s);
		const out = new Uint8Array(bin.length);
		for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
		return out;
	}
	return new Uint8Array(Buffer.from(s, 'base64'));
}

export function decodeArrays(doc: SplitJSON): Arrays {
	const out: Arrays = {};
	for (const [name, a] of Object.entries(doc.arrays)) {
		const bytes = b64bytes(a.b64);
		const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
		const data =
			a.dtype === 'f32' ? new Float32Array(buf) : a.dtype === 'f64' ? new Float64Array(buf) : a.dtype === 'i16' ? new Int16Array(buf) : new Uint8Array(buf);
		out[name] = { dtype: a.dtype, shape: a.shape, data };
	}
	return out;
}

/** One instance (index i) of a split, unpadded: every array with the instance axis removed and cut to n items. */
export function instanceArrays(d: Arrays, i: number): { n: number; arrays: Arrays } {
	const n = d.n.data[i];
	const arrays: Arrays = {};
	for (const [name, a] of Object.entries(d)) {
		if (name === 'n') continue;
		const shape = a.shape.slice(1);
		const per = shape.reduce((x, y) => x * y, 1);
		const src = a.data.subarray(i * per, (i + 1) * per);
		// cut the item axes (size M) to n: per-pair arrays have two (A, adj and the pair outputs), inputs x_* one (a
		// category axis may follow), per-input values none
		const M = shape[0];
		const nItemAxes = shape.length === 0 ? 0 : shape.length >= 2 && !name.startsWith('x_') ? 2 : 1;
		const newShape = shape.map((s, k) => (k < nItemAxes ? n : s));
		const Ctor = src.constructor as new (len: number) => typeof src;
		const out = new Ctor(newShape.reduce((x, y) => x * y, 1));
		if (nItemAxes === 0) out.set(src);
		else if (nItemAxes === 1) {
			const inner = shape.slice(1).reduce((x, y) => x * y, 1);
			out.set(src.subarray(0, n * inner));
		} else {
			const inner = shape.slice(2).reduce((x, y) => x * y, 1);
			for (let u = 0; u < n; u++) out.set(src.subarray(u * M * inner, u * M * inner + n * inner), u * n * inner);
		}
		arrays[name] = { dtype: a.dtype, shape: newShape, data: out };
	}
	return { n, arrays };
}

/** The model's inputs for one unpadded instance (arrays named as in the split files). */
export function modelInputs(model: ModelConfig['model'], n: number, a: Arrays): Inputs {
	const block = model.block;
	const selfLoops = !!model.self_loops;
	if ('x_pos' in a) {
		const perN = new Set(model.scale_by_n ?? []);
		const num = (k: string, i: number) => (perN.has(k.slice(2)) ? a[k].data[i] / Math.max(n, 1) : a[k].data[i]);
		const keys = Object.keys(a)
			.filter((k) => k.startsWith('x_') && k !== 'x_pos')
			.sort();
		const scal = keys.filter((k) => a[k].shape.length === 1 && a[k].dtype === 'f64');
		const msk = keys.filter((k) => a[k].shape.length === 1 && a[k].dtype !== 'f64');
		const cat = keys.filter((k) => a[k].shape.length === 2);
		const graph = keys.filter((k) => a[k].shape.length === 0);
		const cols: ((i: number) => number[])[] = [(i) => [a.x_pos.data[i]]];
		for (const k of [...scal, ...msk]) cols.push((i) => [num(k, i)]);
		for (const k of cat) {
			const C = a[k].shape[1];
			cols.push((i) => Array.from(a[k].data.subarray(i * C, (i + 1) * C)));
		}
		const rows = Array.from({ length: n }, (_, i) => cols.flatMap((c) => c(i)));
		const F = rows[0].length;
		const f = new Float32Array(n * F);
		rows.forEach((r, i) => f.set(r, i * F));
		const g = Float32Array.from(graph, (k) => a[k].data[0]);
		if (block !== 'compare_first') return { n, f, g };
		// compare_first: the numbers (pos, the item numbers, the per-input numbers) move to registers
		const S = scal.length;
		const T = 1 + S + g.length;
		const reg = new Float32Array(T * n);
		for (let r = 0; r < 1 + S; r++) for (let i = 0; i < n; i++) reg[r * n + i] = f[i * F + r];
		for (let q = 0; q < g.length; q++) for (let i = 0; i < n; i++) reg[(1 + S + q) * n + i] = g[q];
		for (let i = 0; i < n; i++) for (let r = 0; r < 1 + S; r++) f[i * F + r] = 0;
		return { n, f, g: new Float32Array(g.length), reg, regRows: T };
	}
	const A = Float32Array.from(a.A.data);
	const adj = Float32Array.from(a.adj.data);
	const pos = Float32Array.from(a.pos.data);
	const s1 = new Float32Array(n);
	if (a.s) s1[a.s.data[0]] = 1;
	// compare-first on a graph task: pos, its only number, rides as the one register row
	if (block === 'compare_first') return { n, A, adj, pos: new Float32Array(n), s1, reg: pos, regRows: 1 };
	if (block !== 'pair_numbers') return { n, A, adj, pos, s1 };
	const edge = new Float32Array(n * n);
	const W = new Float32Array(n * n).fill(BIG);
	for (let u = 0; u < n; u++)
		for (let v = 0; v < n; v++)
			if (u !== v && adj[u * n + v] > 0) {
				edge[u * n + v] = 1;
				W[u * n + v] = A[u * n + v];
			}
	// self loops (Floyd-Warshall): a loop that carries a weight is an edge
	if (selfLoops)
		for (let u = 0; u < n; u++)
			if (adj[u * n + u] > 0 && A[u * n + u] !== 0) {
				edge[u * n + u] = 1;
				W[u * n + u] = A[u * n + u];
			}
	return { n, A: edge, adj, pos: new Float32Array(n), s1, reg: pos, regRows: 1, W };
}
