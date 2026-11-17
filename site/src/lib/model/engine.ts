/**
 * The looped pair block in plain TypeScript (float32 storage, float64 accumulation), one input at a time.
 * A line-by-line port of witness_clrs/model.py (forward only). Arrays are pair-major: x[(u * n + v) * C + c].
 *
 * Every bit the block hands on is a hard comparison; forward, the straight-through terms of the Python code cancel
 * exactly, so the bits here are the bits there. Real-valued steps (linear maps, norm, gelu) can differ from XLA in the
 * last float bit; the parity test counts how often that changes a prediction.
 */
import type { Tensor } from './safetensors';

export type Block = 'census' | 'cold' | 'compare_first' | 'pair_numbers';
export type Kind = 'node' | 'one' | 'edge' | 'nmask' | 'emask' | 'ecat' | 'gmask';

export interface ModelConfig {
	name: string;
	task: string;
	task_name: string;
	block: Block;
	model: {
		block: Block;
		width: number;
		mlp: number;
		edge_mlp: number;
		weight_scale?: 'graph';
		registers?: number[][];
		arithmetic?: 'plus' | 'max';
		/** pair_numbers: the register table ('paths' when absent) and whether weighted loops are edges */
		table?: 'paths' | 'via' | 'hops';
		self_loops?: boolean;
		/** item numbers divided by the input's own n (task scheduling's deadlines, in the units of the positions) */
		scale_by_n?: string[];
		/** compare_first: false when the bits are not kept (both halves recomputed from all the bits each loop) */
		kept_bits?: boolean;
		/** census: a task read after each input's own n loops (the shared block's depth-first search) */
		read?: { rule: 'n'; compile?: string };
	};
	features: { F: number; G: number } | null;
	outputs: { name: string; kind: Kind }[];
	read: { rule: 'fixed' | 'rest'; loops?: number };
	parameters: number;
	scores: Record<string, number>;
	loops_used?: Record<string, { median: number; max: number; rested: number }>;
}

/** The model's inputs for one instance of n items (no padding): what witness_clrs.data.model_inputs makes, B = 1. */
export interface Inputs {
	n: number;
	/** graph tasks */
	A?: Float32Array; // [n, n]; pair_numbers: the fact 'there is an edge'
	adj?: Float32Array; // [n, n]
	pos?: Float32Array; // [n]; pair_numbers: 0
	s1?: Float32Array; // [n] source marker
	W?: Float32Array; // pair_numbers: [n, n] edge weights, BIG where there is no edge
	/** item tasks */
	f?: Float32Array; // [n, F]
	g?: Float32Array; // [G]
	/** registers (compare_first: [T, n] input numbers by row; pair_numbers: [1, n] positions) */
	reg?: Float32Array;
	regRows?: number;
}

export const BIG = Math.fround(1e30);
const NPREG = 8;
const PREG_TABLE = ['grow', 'edge', 'fwd', 'bwd', 'fwd', 'bwd', 'fwd', 'bwd'] as const;
export type RegKind = 'grow' | 'edge' | 'fwd' | 'bwd' | 'via' | 'hgrow' | 'hfwd' | 'hbwd';
export const REG_TABLES: Record<'paths' | 'via' | 'hops', readonly RegKind[]> = {
	paths: PREG_TABLE,
	via: ['grow', 'edge', 'via', 'via', 'fwd', 'bwd', 'fwd', 'bwd'],
	hops: ['grow', 'edge', 'hgrow', 'hbwd', 'hfwd', 'hbwd', 'fwd', 'bwd']
};
/** pair_numbers with the 'via' or 'hops' table or self loops: the loop state carries H and the via hits */
export const pnV2 = (m: ModelConfig['model']) => m.block === 'pair_numbers' && ((m.table ?? 'paths') !== 'paths' || !!m.self_loops);
export const regTable = (m: ModelConfig['model']) => REG_TABLES[m.table ?? 'paths'];
const fr = Math.fround;

// ------------------------------------------------------------------------------------------------ elementwise
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
const softplus = (x: number) => Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));

/** y[r, j] = sum_k x[r, k] W[k, colStart + j] + b[colStart + j], j < cols. W is [cin, cout] row-major. */
export function lin(x: Float32Array, rows: number, cin: number, W: Tensor, b: Tensor | null, colStart = 0, cols?: number): Float32Array {
	const cout = W.shape[1];
	const nc = cols ?? cout - colStart;
	const w = W.data;
	const y = new Float32Array(rows * nc);
	const acc = new Float64Array(nc);
	for (let r = 0; r < rows; r++) {
		acc.fill(0);
		const xr = r * cin;
		for (let k = 0; k < cin; k++) {
			const a = x[xr + k];
			if (a === 0) continue;
			const wk = k * cout + colStart;
			for (let j = 0; j < nc; j++) acc[j] += a * w[wk + j];
		}
		const yr = r * nc;
		if (b) for (let j = 0; j < nc; j++) y[yr + j] = acc[j] + b.data[colStart + j];
		else for (let j = 0; j < nc; j++) y[yr + j] = acc[j];
	}
	return y;
}

/** Normalise every row of x [rows, C] over its C channels. */
export function norm(x: Float32Array, rows: number, C: number): Float32Array {
	const y = new Float32Array(rows * C);
	for (let r = 0; r < rows; r++) {
		const o = r * C;
		let s = 0;
		for (let c = 0; c < C; c++) s += x[o + c];
		const mean = fr(s / C);
		let v = 0;
		for (let c = 0; c < C; c++) {
			const d = x[o + c] - mean;
			v += d * d;
		}
		const inv = 1 / Math.sqrt(fr(v / C) + 1e-6);
		for (let c = 0; c < C; c++) y[o + c] = (x[o + c] - mean) * inv;
	}
	return y;
}

function mlpUpdate(p: Params, z: Float32Array, P: number, D: number): Float32Array {
	const M = p['blk/f1'].shape[1];
	const h = lin(norm(z, P, D), P, D, p['blk/f1'], p['blk/bf1']);
	for (let i = 0; i < h.length; i++) h[i] = gelu(h[i]);
	const u = lin(h, P, M, p['blk/f2'], p['blk/bf2']);
	const out = new Float32Array(z.length);
	for (let i = 0; i < z.length; i++) out[i] = z[i] + u[i];
	return out;
}

export type Params = Record<string, Tensor>;

// ------------------------------------------------------------------------------------------------ embedding
function graphScale(A: Float32Array, n: number): number {
	const l1: number[] = [];
	for (let u = 0; u < n; u++) {
		let best = Infinity;
		for (let v = 0; v < n; v++) {
			if (u === v) continue;
			const a = A[u * n + v] > 0 ? A[u * n + v] : Infinity;
			const b = A[v * n + u] > 0 ? A[v * n + u] : Infinity;
			best = Math.min(best, a, b);
		}
		if (Number.isFinite(best)) l1.push(best);
	}
	const median = (v: number[]) => {
		const s = [...v].sort((a, b) => a - b);
		const c = s.length;
		return fr((fr(s[Math.max(Math.floor((c - 1) / 2), 0)] + s[Math.min(Math.floor(c / 2), n - 1)])) / 2);
	};
	if (l1.length) return median(l1);
	const d: number[] = [];
	for (let u = 0; u < n; u++) if (A[u * n + u] > 0) d.push(A[u * n + u]);
	return d.length ? median(d) : 1;
}

function embedGraph(p: Params, x: Inputs, D: number): Float32Array {
	const n = x.n;
	const P = n * n;
	const A = x.A!;
	const adj = x.adj!;
	const rows = p['e1'].shape[0];
	const NEf = rows;
	const e = new Float32Array(P * NEf);
	if (rows === 11) {
		const s = graphScale(A, n);
		const feat = (a: number): [number, number, number] => {
			if (!(a > 0)) return [0, 0, 0];
			const xx = fr(a / s);
			const lx = fr(Math.log(xx));
			return [fr(xx / fr(1 + fr(xx / 16))), Math.min(4, Math.max(-4, lx)), fr(Math.exp(-xx))];
		};
		for (let u = 0; u < n; u++)
			for (let v = 0; v < n; v++) {
				const i = (u * n + v) * NEf;
				const f = feat(A[u * n + v]);
				const ft = feat(A[v * n + u]);
				e.set([f[0], ft[0], A[u * n + v] !== 0 ? 1 : 0, A[v * n + u] !== 0 ? 1 : 0, adj[u * n + v], adj[v * n + u], u === v ? 1 : 0, f[1], ft[1], f[2], ft[2]], i);
			}
	} else {
		for (let u = 0; u < n; u++)
			for (let v = 0; v < n; v++) {
				const i = (u * n + v) * NEf;
				e.set([A[u * n + v], A[v * n + u], A[u * n + v] !== 0 ? 1 : 0, A[v * n + u] !== 0 ? 1 : 0, adj[u * n + v], adj[v * n + u], u === v ? 1 : 0], i);
			}
	}
	const W = p['e1'].shape[1];
	const h1 = lin(e, P, NEf, p['e1'], p['eb1']);
	for (let i = 0; i < h1.length; i++) h1[i] = gelu(h1[i]);
	const h = lin(h1, P, W, p['e2'], null);
	const h2 = lin(e, P, NEf, p['E'], p['eb']);
	const fn = new Float32Array(n * 2);
	for (let u = 0; u < n; u++) {
		fn[u * 2] = x.pos![u];
		fn[u * 2 + 1] = x.s1![u];
	}
	const r = lin(fn, n, 2, p['R'], null);
	const c = lin(fn, n, 2, p['C'], null);
	const z = new Float32Array(P * D);
	for (let u = 0; u < n; u++)
		for (let v = 0; v < n; v++) {
			const i = (u * n + v) * D;
			for (let d = 0; d < D; d++) z[i + d] = fr(fr(fr(h[i + d] + h2[i + d]) + r[u * D + d]) + c[v * D + d]);
		}
	return z;
}

function embedItems(p: Params, x: Inputs, D: number): Float32Array {
	const n = x.n;
	const P = n * n;
	const G = x.g!.length;
	const ne = 2 + G;
	const e = new Float32Array(P * ne);
	for (let u = 0; u < n; u++)
		for (let v = 0; v < n; v++) {
			const i = (u * n + v) * ne;
			e[i] = 1;
			e[i + 1] = u === v ? 1 : 0;
			for (let k = 0; k < G; k++) e[i + 2 + k] = x.g![k];
		}
	const W = p['in30/e1'].shape[1];
	const h1 = lin(e, P, ne, p['in30/e1'], p['in30/eb1']);
	for (let i = 0; i < h1.length; i++) h1[i] = gelu(h1[i]);
	const h = lin(h1, P, W, p['in30/e2'], null);
	const h2 = lin(e, P, ne, p['in30/E'], p['in30/eb']);
	const F = p['in30/n1'].shape[0];
	const g1 = lin(x.f!, n, F, p['in30/n1'], p['in30/nb1']);
	const uu = new Float32Array(n * (F + W));
	for (let i = 0; i < n; i++) {
		for (let k = 0; k < F; k++) uu[i * (F + W) + k] = x.f![i * F + k];
		for (let k = 0; k < W; k++) uu[i * (F + W) + F + k] = gelu(g1[i * W + k]);
	}
	const r = lin(uu, n, F + W, p['in30/R'], null);
	const c = lin(uu, n, F + W, p['in30/C'], null);
	const z = new Float32Array(P * D);
	for (let u = 0; u < n; u++)
		for (let v = 0; v < n; v++) {
			const i = (u * n + v) * D;
			for (let d = 0; d < D; d++) z[i + d] = fr(fr(fr(h[i + d] + h2[i + d]) + r[u * D + d]) + c[v * D + d]);
		}
	return z;
}

export function embed(p: Params, x: Inputs, D: number): Float32Array {
	return x.f ? embedItems(p, x, D) : embedGraph(p, x, D);
}

// ------------------------------------------------------------------------------------------------ the two halves
/** What a count channel pair saw, kept for the page: per pair (u, v) and count channel c, N_c(u, v). */
export interface Trace {
	/** tests of the count half: a[(u n + k) h + c] = [xa_c(u, k) > 0], b[(k n + v) h + c] = [xc_c(k, v) > 0] (after fixed tests) */
	testA: Uint8Array;
	testB: Uint8Array;
	/** witness counts at the level that decides (census: all witnesses), [P, h] */
	counts: Uint16Array;
	/** best half: the best witness k per pair and channel ([P, h], 255 = none) */
	best: Uint8Array;
	/** register channel pairs (compare_first/pair_numbers): the leading value per pair and channel pair, NaN for facts */
	lead?: Float32Array;
}

function countCompare(xa: Float32Array, xc: Float32Array, n: number, h: number, D: number, aOff: number, chans: number[], y: Float32Array, yOff: number, trace: Trace) {
	// chans: the count channels (pairs of consecutive entries) to run, columns of xa/xc (stride D) starting at aOff
	const P = n * n;
	const C = chans.length;
	const N = new Uint16Array(P * C);
	for (let u = 0; u < n; u++)
		for (let k = 0; k < n; k++) {
			const ia = (u * n + k) * D + aOff;
			for (let v = 0; v < n; v++) {
				const ic = (k * n + v) * D + aOff;
				const o = (u * n + v) * C;
				for (let q = 0; q < C; q++) {
					const c = chans[q];
					if (xa[ia + c] > 0 && xc[ic + c] > 0) N[o + q]++;
				}
			}
		}
	for (let i = 0; i < P; i++)
		for (let q = 0; q < C; q += 2) {
			const d = N[i * C + q] - N[i * C + q + 1];
			y[i * D + yOff + chans[q]] = d > 0 ? 1 : 0;
			y[i * D + yOff + chans[q + 1]] = d === 0 ? 1 : 0;
			trace.counts[i * h + chans[q]] = N[i * C + q];
			trace.counts[i * h + chans[q + 1]] = N[i * C + q + 1];
		}
}

function bestHalf(xa: Float32Array, xc: Float32Array, n: number, h: number, D: number, theta: Float32Array, y: Float32Array, bits: boolean, trace: Trace) {
	const P = n * n;
	const ca = new Float32Array(P * h);
	const cc = new Float32Array(P * h);
	for (let i = 0; i < P; i++)
		for (let c = 0; c < h; c++) {
			ca[i * h + c] = softplus(-xa[i * D + h + c]);
			cc[i * h + c] = softplus(-xc[i * D + h + c]);
		}
	const best = new Float64Array(h);
	const arg = new Int32Array(h);
	const any = new Uint8Array(h);
	for (let u = 0; u < n; u++)
		for (let v = 0; v < n; v++) {
			best.fill(Infinity);
			arg.fill(255);
			any.fill(0);
			for (let k = 0; k < n; k++) {
				const ia = (u * n + k) * h;
				const ic = (k * n + v) * h;
				for (let c = 0; c < h; c++) {
					const a = ca[ia + c];
					const b = cc[ic + c];
					const cost = fr(Math.max(a, b) + fr(theta[c] * Math.min(a, b)));
					if (cost < best[c]) {
						best[c] = cost;
						arg[c] = k;
					}
					if (bits && xa[(u * n + k) * D + h + c] > 0 && xc[(k * n + v) * D + h + c] > 0) any[c] = 1;
				}
			}
			const i = u * n + v;
			for (let c = 0; c < h; c++) {
				y[i * D + h + c] = bits ? any[c] : Math.exp(-best[c]);
				trace.best[i * h + c] = arg[c];
			}
		}
}

// ------------------------------------------------------------------------------------------------ registers
export function registerWiring(h: number, kinds: number[][]): { wire: Int32Array; occ: Int32Array } {
	const P = h / 2;
	const combos: [number, number][] = [];
	for (const g of kinds) for (let i = 0; i < g.length; i++) for (let j = i; j < g.length; j++) combos.push([g[i], g[j]]);
	const nf = combos.length ? Math.floor(P / 2) : P;
	const wire = new Int32Array(h);
	const occ = new Int32Array(P).fill(-1);
	for (let i = nf; i < P; i++) {
		const c = combos[(i - nf) % combos.length];
		wire[2 * i] = c[0];
		wire[2 * i + 1] = c[1];
		occ[i] = Math.floor((i - nf) / combos.length);
	}
	return { wire, occ };
}

export function ownFlags(h: number, kinds: number[][]): Int32Array {
	const { occ } = registerWiring(h, kinds);
	const f = new Int32Array(h);
	for (let i = 0; i < h / 2; i++) {
		if (occ[i] === 0) [f[2 * i], f[2 * i + 1]] = [1, 2];
		if (occ[i] === 1) [f[2 * i], f[2 * i + 1]] = [2, 1];
	}
	return f;
}

/**
 * The count half with registers (lead_compare): channels 0..wire.length-1 of the count half, columns of la/lc
 * (stride D, offset 0). Fact pairs (wire 0) count witnesses; register pairs compare the register values of their
 * admitted witnesses at the joint leading level. own: fixed tests (1: the witness is u, 2: the witness is v).
 * Per channel the tests are laid out as two n x n bit matrices, the right one transposed, so the scan over the middle
 * item k is contiguous. A register pair is one pass: the lowest admitted value so far and how many reach it.
 */
export function leadCompare(la: Float32Array, lc: Float32Array, n: number, h: number, D: number, reg: Float32Array, wire: Int32Array, own: Int32Array, y: Float32Array, trace: Trace, facts?: { y: Float32Array; counts: Uint32Array }) {
	const L = wire.length;
	const nn = n * n;
	const tA = trace.testA;
	const tB = trace.testB;
	// left test a_c(r, s) and right test b_c(r, s) (the witness r for a pair ending at s), for the trace
	for (let r = 0; r < n; r++)
		for (let s = 0; s < n; s++) {
			const i = r * n + s;
			const o = i * D;
			const t = i * h;
			for (let c = 0; c < L; c++) {
				const ow = own[c];
				tA[t + c] = ow === 1 ? (r === s ? 1 : 0) : ow === 2 ? 1 : la[o + c] > 0 ? 1 : 0;
				tB[t + c] = ow === 1 ? 1 : ow === 2 ? (r === s ? 1 : 0) : lc[o + c] > 0 ? 1 : 0;
			}
		}
	// per channel: A_c[u n + k] = a_c(u, k); BT_c[v n + k] = b_c(k, v)
	const A = new Uint8Array(L * nn);
	const BT = new Uint8Array(L * nn);
	for (let r = 0; r < n; r++)
		for (let s = 0; s < n; s++) {
			const t = (r * n + s) * h;
			for (let c = 0; c < L; c++) {
				A[c * nn + r * n + s] = tA[t + c];
				BT[c * nn + s * n + r] = tB[t + c];
			}
		}
	const rv = (row: number) => {
		const v = new Float32Array(n);
		if (row > 0) for (let k = 0; k < n; k++) v[k] = reg[(row - 1) * n + k];
		return v;
	};
	for (let q = 0; q < L; q += 2) {
		const c1 = q;
		const c2 = q + 1;
		const a1 = c1 * nn;
		const a2 = c2 * nn;
		const fact = wire[c1] === 0 && wire[c2] === 0;
		if (fact && facts) {
			// counted elsewhere (the GPU) from the same tests
			for (let i = 0; i < nn; i++) {
				y[i * D + c1] = facts.y[i * D + c1];
				y[i * D + c2] = facts.y[i * D + c2];
				trace.counts[i * h + c1] = facts.counts[i * h + c1];
				trace.counts[i * h + c2] = facts.counts[i * h + c2];
			}
			continue;
		}
		const v1 = fact ? null : rv(wire[c1]);
		const v2 = fact ? null : rv(wire[c2]);
		for (let u = 0; u < n; u++)
			for (let v = 0; v < n; v++) {
				const ou = u * n;
				const ov = v * n;
				let n1 = 0;
				let n2 = 0;
				let Es = Infinity;
				if (fact) {
					for (let k = 0; k < n; k++) {
						n1 += A[a1 + ou + k] & BT[a1 + ov + k];
						n2 += A[a2 + ou + k] & BT[a2 + ov + k];
					}
				} else {
					for (let k = 0; k < n; k++) {
						if (A[a1 + ou + k] & BT[a1 + ov + k]) {
							const x = v1![k];
							if (x < Es) {
								Es = x;
								n1 = 1;
								n2 = 0;
							} else if (x === Es) n1++;
						}
						if (A[a2 + ou + k] & BT[a2 + ov + k]) {
							const x = v2![k];
							if (x < Es) {
								Es = x;
								n1 = 0;
								n2 = 1;
							} else if (x === Es) n2++;
						}
					}
				}
				const i = u * n + v;
				y[i * D + c1] = n1 > n2 ? 1 : 0;
				y[i * D + c2] = n1 === n2 ? 1 : 0;
				trace.counts[i * h + c1] = n1;
				trace.counts[i * h + c2] = n2;
				if (!fact && trace.lead) trace.lead[i * (h / 2) + q / 2] = Es;
			}
	}
}

// ------------------------------------------------------------------------------------------------ pair numbers
const present = (v: number) => v < BIG / 2;

export function pnLayout(h: number, table: readonly RegKind[] = PREG_TABLE) {
	const P = h / 2;
	const NNODE = Math.floor(P / 4);
	const base = 2 * (P - NPREG);
	const wire = new Int32Array(base);
	for (let i = 2 * (P - NPREG - NNODE); i < base; i++) wire[i] = 1;
	const own = new Int32Array(base);
	const s = 2 * (P - NPREG - NNODE);
	own.set([1, 2, 2, 1], s);
	const ownbit = new Uint8Array(h);
	for (let i = 0; i < base; i++) ownbit[i] = own[i] > 0 ? 1 : 0;
	table.forEach((t, i) => {
		if (t === 'edge') ownbit[base + 2 * i] = ownbit[base + 2 * i + 1] = 1;
	});
	return { base, wire, own, ownbit };
}

/** What the register pairs build in one read: the new R, the hop candidate (H or better; null without H), and the via
 * hits of each via pair, [u, k, v] (1 where k is admitted and R(u, k) (+) W(k, v) equals R(u, v)). */
export interface RegOut {
	R: Float32Array;
	H: Float32Array | null;
	hits: Uint8Array[];
}

/**
 * The 8 register channel pairs: bits into y (columns base..h-1), the new R (and H), the via hits. la/lc: count-half
 * columns (stride D). Sums in float32 exactly as the Python block forms them ((z + a) + c with the identity 0 dropping
 * out); 'max' with the identity -BIG dropping out.
 */
export function regPairs(la: Float32Array, lc: Float32Array, n: number, h: number, D: number, base: number, W: Float32Array, R: Float32Array, plus: boolean, y: Float32Array, trace: Trace, table: readonly RegKind[] = PREG_TABLE, H: Float32Array | null = null): RegOut {
	const nn = n * n;
	const Rn = R.slice();
	const Hc = H ? H.slice() : null;
	const RT = new Float32Array(nn); // RT[a n + b] = R[b n + a]
	const WT = new Float32Array(nn);
	const HT = new Float32Array(H ? nn : 0);
	for (let a = 0; a < n; a++)
		for (let b = 0; b < n; b++) {
			RT[a * n + b] = R[b * n + a];
			WT[a * n + b] = W[b * n + a];
			if (H) HT[a * n + b] = H[b * n + a];
		}
	const hits: Uint8Array[] = [];
	const A = new Uint8Array(nn);
	const BT = new Uint8Array(nn);
	for (let i = 0; i < table.length; i++) {
		const t = table[i];
		const c1 = base + 2 * i;
		for (let r = 0; r < n; r++)
			for (let s = 0; s < n; s++) {
				const ta = la[(r * n + s) * D + c1] > 0 ? 1 : 0;
				const tb = lc[(r * n + s) * D + c1] > 0 ? 1 : 0;
				trace.testA[(r * n + s) * h + c1] = ta;
				trace.testB[(r * n + s) * h + c1] = tb;
				A[r * n + s] = ta;
				BT[s * n + r] = tb;
			}
		const hit = t === 'via' ? new Uint8Array(nn * n) : null;
		if (hit) hits.push(hit);
		for (let u = 0; u < n; u++)
			for (let v = 0; v < n; v++) {
				const p = u * n + v;
				const ou = u * n;
				const ov = v * n;
				let less = false;
				let level = false;
				let cnt = 0;
				if (t === 'grow' || t === 'hgrow') {
					// grow: R(u, k) + W(k, v) ('plus') or max(R(u, k), R(k, v)) ('max'); hgrow: H(u, k) + H(k, v)
					const hop = t === 'hgrow';
					const X = hop ? H! : R;
					let L1 = BIG;
					for (let k = 0; k < n; k++) {
						if (!(A[ou + k] & BT[ov + k])) continue;
						cnt++;
						const val = hop ? fr(H![ou + k] + HT[ov + k]) : plus ? fr(R[ou + k] + WT[ov + k]) : Math.max(R[ou + k], RT[ov + k]);
						if (val < BIG / 2 && val < L1) L1 = val;
					}
					if (hop) Hc![p] = Math.min(Hc![p], L1);
					else Rn[p] = Math.min(Rn[p], L1);
					const own = X[p];
					const L2 = own < BIG / 2 ? own : BIG;
					less = L1 < L2;
					level = L1 === L2 && L1 < BIG / 2;
					if (trace.lead) trace.lead[p * (h / 2) + c1 / 2] = L1;
				} else if (t === 'edge') {
					const a = R[p] < BIG / 2 ? R[p] : BIG;
					const b = W[p] < BIG / 2 ? W[p] : BIG;
					less = a < b;
					level = a === b && a < BIG / 2;
				} else if (t === 'via') {
					// per admitted k: R(u, k) (+) W(k, v) against R(u, v); a hit where they are equal
					const V2 = R[p];
					for (let k = 0; k < n; k++) {
						if (!(A[ou + k] & BT[ov + k])) continue;
						cnt++;
						const V1 = plus ? fr(R[ou + k] + WT[ov + k]) : Math.max(R[ou + k], WT[ov + k]);
						if (V1 < V2 && V1 < BIG / 2) less = true;
						if (V1 === V2 && V1 < BIG / 2 && V2 < BIG / 2) {
							level = true;
							hit![(u * n + k) * n + v] = 1;
						}
					}
				} else if (t === 'hfwd' || t === 'hbwd') {
					// hfwd: H(k, u) against H(k, v); hbwd: H(k, v) against H(k, u)
					const f = t === 'hfwd';
					for (let k = 0; k < n; k++) {
						if (!(A[ou + k] & BT[ov + k])) continue;
						cnt++;
						const V1 = f ? HT[ou + k] : HT[ov + k];
						const V2 = f ? HT[ov + k] : HT[ou + k];
						if (V1 < V2 && V1 < BIG / 2) less = true;
						if (V1 === V2 && V1 < BIG / 2 && V2 < BIG / 2) level = true;
					}
				} else {
					const fwd = t === 'fwd';
					const w = fwd ? W[p] : W[v * n + u];
					for (let k = 0; k < n; k++) {
						if (!(A[ou + k] & BT[ov + k])) continue;
						cnt++;
						// fwd: W(u, v) (+) R(k, u) against R(k, v); bwd: W(v, u) (+) R(k, v) against R(k, u)
						const rku = RT[ou + k];
						const rkv = RT[ov + k];
						const V1 = plus ? fr(w + (fwd ? rku : rkv)) : Math.max(w, fwd ? rku : rkv);
						const V2 = fwd ? rkv : rku;
						if (V1 < V2 && V1 < BIG / 2) less = true;
						if (V1 === V2 && V1 < BIG / 2 && V2 < BIG / 2) level = true;
					}
				}
				y[p * D + c1] = less ? 1 : 0;
				y[p * D + c1 + 1] = level ? 1 : 0;
				trace.counts[p * h + c1] = cnt;
				trace.counts[p * h + c1 + 1] = cnt;
			}
	}
	return { R: Rn, H: Hc, hits };
}

// ------------------------------------------------------------------------------------------------ the engine
export interface State {
	loop: number;
	/** census: the pair state z; resting blocks: the bits y */
	s: Float32Array;
	/** pair_numbers: the built numbers R [n, n] (and H, the hops, with the 'via'/'hops' tables or self loops) */
	R?: Float32Array;
	H?: Float32Array;
	/** pair_numbers with via pairs: the via hits of the loop that made this state (the pointer read-out adds them) */
	hits?: Uint8Array[];
	/** entries of the bits (and R) that changed in this loop (resting blocks) */
	changed: number;
	trace: Trace | null;
}

export class Engine {
	readonly D: number;
	readonly h: number;
	readonly block: Block;
	protected e: Float32Array | null = null;
	x: Inputs | null = null;
	wire: Int32Array | null = null;
	own: Int32Array | null = null;
	theta: Float32Array;
	/** pair_numbers: the register table, and whether the state carries H and the via hits */
	readonly table: readonly RegKind[];
	readonly v2: boolean;
	/** compare_first: whether the bits are kept (otherwise both halves read the state rebuilt from all the bits) */
	readonly kept: boolean;

	constructor(
		readonly p: Params,
		readonly config: ModelConfig
	) {
		this.D = config.model.width;
		this.h = p['blk/th_t'].shape[0];
		this.block = config.model.block;
		const th = p['blk/th_t'].data;
		this.theta = new Float32Array(th.length);
		for (let c = 0; c < th.length; c++) this.theta[c] = fr(th[c] + fr(Math.min(1, Math.max(0, th[c])) - th[c]));
		if (this.block === 'compare_first') {
			this.wire = registerWiring(this.h, config.model.registers!).wire;
			this.own = ownFlags(this.h, config.model.registers!);
		}
		this.table = regTable(config.model);
		this.v2 = pnV2(config.model);
		this.kept = config.model.kept_bits ?? true;
	}

	/** bits that depend on no learned test (compare_first: the own comparisons; pair_numbers: the edge pair) */
	ownbits(): number[] {
		return this.block === 'compare_first' ? Array.from(this.own!, (o) => (o > 0 ? 1 : 0)) : Array.from(pnLayout(this.h, this.table).ownbit);
	}

	private newTrace(n: number): Trace {
		const P = n * n;
		return {
			testA: new Uint8Array(P * this.h),
			testB: new Uint8Array(P * this.h),
			counts: new Uint16Array(P * this.h),
			best: new Uint8Array(P * this.h),
			lead: this.block === 'census' ? undefined : new Float32Array(P * (this.h / 2)).fill(NaN)
		};
	}

	/** The rebuilt state of a resting block: e + o(y) + MLP. */
	private rebuild(y: Float32Array, n: number): Float32Array {
		const P = n * n;
		const D = this.D;
		const u = lin(y, P, D, this.p['blk/o'], this.p['blk/bo']);
		const s = new Float32Array(P * D);
		for (let i = 0; i < s.length; i++) s[i] = this.e![i] + u[i];
		return mlpUpdate(this.p, s, P, D);
	}

	/** One read of a resting block's count half under the state z -> bits into y (count columns); returns what the
	 * register pairs built (pair_numbers). */
	private countHalf(z: Float32Array, n: number, y: Float32Array, trace: Trace, R?: Float32Array, H?: Float32Array): RegOut | undefined {
		const P = n * n;
		const { D, h } = this;
		const t = norm(z, P, D);
		const la = lin(t, P, D, this.p['blk/a'], this.p['blk/ba'], 0, h);
		const lc = lin(t, P, D, this.p['blk/b'], this.p['blk/bb'], 0, h);
		// la/lc have stride h here; re-stride to D so the helpers index uniformly
		const A = restride(la, P, h, D);
		const C = restride(lc, P, h, D);
		if (this.block === 'compare_first') {
			leadCompare(A, C, n, h, D, this.x!.reg!, this.wire!, this.own!, y, trace);
			return undefined;
		}
		const { base, wire, own } = pnLayout(h, this.table);
		leadCompare(A, C, n, h, D, this.x!.reg!, wire, own, y, trace);
		return regPairs(A, C, n, h, D, base, this.x!.W!, R!, this.config.model.arithmetic === 'plus', y, trace, this.table, H ?? null);
	}

	private existsHalf(z: Float32Array, n: number, y: Float32Array, trace: Trace) {
		const P = n * n;
		const { D, h } = this;
		const t = norm(z, P, D);
		const ua = restride(lin(t, P, D, this.p['blk/a'], this.p['blk/ba'], h, h), P, h, D, h);
		const uc = restride(lin(t, P, D, this.p['blk/b'], this.p['blk/bb'], h, h), P, h, D, h);
		bestHalf(ua, uc, n, h, D, this.theta, y, true, trace);
	}

	start(x: Inputs): State {
		this.x = x;
		const n = x.n;
		const P = n * n;
		const D = this.D;
		const h = this.h;
		this.e = embed(this.p, x, D);
		if (this.block === 'census') return { loop: 0, s: this.e.slice(), changed: P * D, trace: null };
		const y = new Float32Array(P * D);
		const trace = this.newTrace(n);
		if (this.block === 'cold' || !this.kept) return { loop: 0, s: y, changed: P * D, trace: null };
		const c0 = new Float32Array(P * D);
		let R: Float32Array | undefined;
		let H: Float32Array | undefined;
		if (this.block === 'pair_numbers') R = startReg(x, this.config.model.arithmetic!);
		if (this.v2) H = this.config.model.table === 'hops' ? startHops(x, R!) : new Float32Array(P);
		const built = this.countHalf(this.rebuild(y, n), n, c0, trace, R, H);
		const ownbit = this.ownbits();
		for (let i = 0; i < P; i++) for (let c = 0; c < h; c++) y[i * D + c] = c0[i * D + c] * ownbit[c];
		return { loop: 0, s: y, R, H, hits: this.v2 ? built!.hits : undefined, changed: P * D, trace: null };
	}

	step(st: State): State {
		const n = this.x!.n;
		const P = n * n;
		const { D, h } = this;
		const trace = this.newTrace(n);
		if (this.block === 'census') {
			const z = st.s;
			const t = norm(z, P, D);
			const xa = lin(t, P, D, this.p['blk/a'], this.p['blk/ba']);
			const xc = lin(t, P, D, this.p['blk/b'], this.p['blk/bb']);
			const y = new Float32Array(P * D);
			const chans = Array.from({ length: h }, (_, c) => c);
			for (let u = 0; u < n; u++)
				for (let k = 0; k < n; k++)
					for (let c = 0; c < h; c++) {
						trace.testA[(u * n + k) * h + c] = xa[(u * n + k) * D + c] > 0 ? 1 : 0;
						trace.testB[(u * n + k) * h + c] = xc[(u * n + k) * D + c] > 0 ? 1 : 0;
					}
			countCompare(xa, xc, n, h, D, 0, chans, y, 0, trace);
			bestHalf(xa, xc, n, h, D, this.theta, y, false, trace);
			const gate = lin(t, P, D, this.p['blk/g'], this.p['blk/bg']);
			const upd = lin(y, P, D, this.p['blk/o'], this.p['blk/bo']);
			const z2 = new Float32Array(P * D);
			for (let i = 0; i < z2.length; i++) z2[i] = z[i] + fr(sigmoid(gate[i]) * upd[i]);
			return { loop: st.loop + 1, s: mlpUpdate(this.p, z2, P, D), changed: P * D, trace };
		}
		const y = st.s;
		if (this.block === 'cold') {
			// both halves read the state rebuilt from the bits: more / same, and [some witness passes both tests]
			const t = norm(this.rebuild(y, n), P, D);
			const xa = lin(t, P, D, this.p['blk/a'], this.p['blk/ba']);
			const xc = lin(t, P, D, this.p['blk/b'], this.p['blk/bb']);
			for (let u = 0; u < n; u++)
				for (let k = 0; k < n; k++)
					for (let c = 0; c < h; c++) {
						trace.testA[(u * n + k) * h + c] = xa[(u * n + k) * D + c] > 0 ? 1 : 0;
						trace.testB[(u * n + k) * h + c] = xc[(u * n + k) * D + c] > 0 ? 1 : 0;
					}
			const y2 = new Float32Array(P * D);
			countCompare(xa, xc, n, h, D, 0, Array.from({ length: h }, (_, c) => c), y2, 0, trace);
			bestHalf(xa, xc, n, h, D, this.theta, y2, true, trace);
			let changed = 0;
			for (let i = 0; i < y2.length; i++) if (y2[i] !== y[i]) changed++;
			return { loop: st.loop + 1, s: y2, changed, trace };
		}
		if (!this.kept) {
			// compare_first without kept bits: both halves from the state rebuilt from all the bits
			const z = this.rebuild(y, n);
			const y2 = new Float32Array(P * D);
			this.existsHalf(z, n, y2, trace);
			this.countHalf(z, n, y2, trace);
			let changed = 0;
			for (let i = 0; i < y2.length; i++) if (y2[i] !== y[i]) changed++;
			return { loop: st.loop + 1, s: y2, changed, trace };
		}
		const ownbit = this.ownbits();
		// kept bits: the exists half only turns on
		const ex = new Float32Array(P * D);
		this.existsHalf(this.rebuild(y, n), n, ex, trace);
		const y2 = new Float32Array(P * D);
		for (let i = 0; i < P; i++) {
			for (let c = 0; c < h; c++) y2[i * D + c] = y[i * D + c] * ownbit[c];
			for (let c = h; c < D; c++) y2[i * D + c] = Math.max(y[i * D + c], ex[i * D + c]);
		}
		const cb = new Float32Array(P * D);
		const built = this.countHalf(this.rebuild(y2, n), n, cb, trace, st.R, st.H);
		for (let i = 0; i < P; i++) for (let c = 0; c < h; c++) y2[i * D + c] = cb[i * D + c];
		return this.finish(st, y2, built, trace);
	}

	/** After a resting block's loop: H's update ('hops': restart from the new R when R moved), and the change count. */
	finish(st: State, y2: Float32Array, built: RegOut | undefined, trace: Trace): State {
		const y = st.s;
		const Rn = built?.R;
		let Hn = st.H;
		if (this.v2 && this.config.model.table === 'hops') {
			let moved = false;
			for (let i = 0; i < Rn!.length && !moved; i++) if (Rn![i] !== st.R![i]) moved = true;
			Hn = moved ? startHops(this.x!, Rn!) : built!.H!;
		}
		let changed = 0;
		for (let i = 0; i < y2.length; i++) if (y2[i] !== y[i]) changed++;
		if (Rn && st.R) for (let i = 0; i < Rn.length; i++) if (Rn[i] !== st.R[i]) changed++;
		if (Hn && st.H) for (let i = 0; i < Hn.length; i++) if (Hn[i] !== st.H[i]) changed++;
		return { loop: st.loop + 1, s: y2, R: Rn, H: Hn, hits: this.v2 ? built!.hits : undefined, changed, trace };
	}

	/** The pair state the read-outs read. */
	state(st: State): Float32Array {
		return this.block === 'census' ? st.s : this.rebuild(st.s, this.x!.n);
	}

	/** Logits and predictions of every output. */
	read(st: State): Record<string, { kind: Kind; pred: Int32Array; logits: Float32Array }> {
		return this.readState(this.state(st), this.x!.n, undefined, st.hits);
	}

	/** Read-outs of a pair state z [n n, D]. edge: the two maps of the pair-pointer read-out, if computed elsewhere;
	 * hits: the via hits of the loop that made the state (the pointer to a third item adds via_c * hit_c). */
	readState(z: Float32Array, n: number, edge?: { Pm: Float32Array; Rm: Float32Array }, hits?: Uint8Array[]): Record<string, { kind: Kind; pred: Int32Array; logits: Float32Array }> {
		const P = n * n;
		const D = this.D;
		const t = norm(z, P, D);
		const out: Record<string, { kind: Kind; pred: Int32Array; logits: Float32Array }> = {};
		const dot = (i: number, w: Float32Array) => {
			let s = 0;
			for (let c = 0; c < D; c++) s += t[i * D + c] * w[c];
			return fr(s);
		};
		const argmax = (l: Float32Array, off: number, len: number) => {
			let b = 0;
			for (let j = 1; j < len; j++) if (l[off + j] > l[off + b]) b = j;
			return b;
		};
		for (const { name, kind } of this.config.outputs) {
			let logits: Float32Array;
			let pred: Int32Array;
			if (kind === 'node') {
				const w = this.p['w_node'].data;
				logits = new Float32Array(P);
				for (let i = 0; i < P; i++) logits[i] = dot(i, w);
				pred = Int32Array.from({ length: n }, (_, i) => argmax(logits, i * n, n));
			} else if (kind === 'one') {
				const w = (this.p[`one30/${name}`] ?? this.p['w_one']).data;
				logits = Float32Array.from({ length: n }, (_, i) => dot(i * n + i, w));
				pred = Int32Array.of(argmax(logits, 0, n));
			} else if (kind === 'nmask') {
				const w = this.p['w_mask'].data;
				const b = this.p['b_mask'].data[0];
				logits = Float32Array.from({ length: n }, (_, i) => fr(dot(i * n + i, w) + b));
				pred = Int32Array.from(logits, (l) => (l > 0 ? 1 : 0));
			} else if (kind === 'emask') {
				const w = this.p['w_mask'].data;
				const b = this.p['b_mask'].data[0];
				const l = Float32Array.from({ length: P }, (_, i) => fr(dot(i, w) + b));
				logits = new Float32Array(P);
				for (let u = 0; u < n; u++) for (let v = 0; v < n; v++) logits[u * n + v] = fr(fr(l[u * n + v] + l[v * n + u]) / 2);
				pred = Int32Array.from(logits, (q) => (q > 0 ? 1 : 0));
			} else if (kind === 'ecat') {
				const W = this.p['ec30/w'];
				const b = this.p['ec30/b'].data;
				const K = W.shape[1];
				logits = new Float32Array(P * K);
				for (let i = 0; i < P; i++)
					for (let q = 0; q < K; q++) {
						let s = 0;
						for (let c = 0; c < D; c++) s += t[i * D + c] * W.data[c * K + q];
						logits[i * K + q] = fr(fr(s) + b[q]);
					}
				pred = Int32Array.from({ length: P }, (_, i) => argmax(logits, i * K, K));
			} else if (kind === 'gmask') {
				const mean = new Float64Array(D);
				for (let i = 0; i < P; i++) for (let c = 0; c < D; c++) mean[c] += t[i * D + c];
				let s = 0;
				const w = this.p['gm30/w'].data;
				for (let c = 0; c < D; c++) s += fr(mean[c] / P) * w[c];
				logits = Float32Array.of(fr(fr(s) + this.p['gm30/b'].data[0]));
				pred = Int32Array.of(logits[0] > 0 ? 1 : 0);
			} else {
				// edge: logits[i, j, k] = P(z_ik) . R(z_kj) / sqrt(D)
				const Pm = edge?.Pm ?? lin(t, P, D, this.p['P'], this.p['bP']);
				const Rm = edge?.Rm ?? lin(t, P, D, this.p['Rr'], this.p['bR']);
				const sc = Math.sqrt(D);
				logits = new Float32Array(P * n);
				for (let i = 0; i < n; i++)
					for (let k = 0; k < n; k++) {
						const pi = (i * n + k) * D;
						for (let j = 0; j < n; j++) {
							const rj = (k * n + j) * D;
							let s = 0;
							for (let c = 0; c < D; c++) s += Pm[pi + c] * Rm[rj + c];
							logits[(i * n + j) * n + k] = fr(fr(s) / sc);
						}
					}
				if (hits && hits.length) {
					// + sum_c via_c * hit_c(i, k, j), summed in float32 as the Python block does
					const via = this.p['via'].data;
					for (let i = 0; i < n; i++)
						for (let k = 0; k < n; k++)
							for (let j = 0; j < n; j++) {
								let a = 0;
								for (let c = 0; c < hits.length; c++) a = fr(a + (hits[c][(i * n + k) * n + j] ? via[c] : 0));
								const q = (i * n + j) * n + k;
								logits[q] = fr(logits[q] + a);
							}
				}
				pred = Int32Array.from({ length: P }, (_, i) => argmax(logits, i * n, n));
			}
			out[name] = { kind, pred, logits };
		}
		return out;
	}
}

export function restride(x: Float32Array, rows: number, cols: number, stride: number, off = 0): Float32Array {
	const y = new Float32Array(rows * stride);
	for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) y[r * stride + off + c] = x[r * cols + c];
	return y;
}

/** R at the start: the edge weights W; on the diagonal a weighted loop's weight, else 0 ('plus') or -BIG ('max'). */
export function startReg(x: Inputs, kind: 'plus' | 'max'): Float32Array {
	const n = x.n;
	const R = x.W!.slice();
	for (let u = 0; u < n; u++) if (!(x.W![u * n + u] < BIG / 2)) R[u * n + u] = kind === 'plus' ? 0 : -BIG;
	return R;
}

/** H at the start (and after R moves): 0 on the diagonal, 1 where the edge is its own best path (R = W), BIG elsewhere. */
export function startHops(x: Inputs, R: Float32Array): Float32Array {
	const n = x.n;
	const W = x.W!;
	const H = new Float32Array(n * n).fill(BIG);
	for (let u = 0; u < n; u++)
		for (let v = 0; v < n; v++) {
			const p = u * n + v;
			if (u === v) H[p] = 0;
			else if (W[p] < BIG / 2 && R[p] === W[p]) H[p] = 1;
		}
	return H;
}
