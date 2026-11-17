/**
 * The block on WebGPU: the dense work (linear maps, norm, MLP, the witness counts and the best-witness join, the
 * pair-pointer read-out) runs in compute shaders; the register comparisons of the cold blocks (compare-first, pair
 * numbers) and the read-outs run on the CPU with engine.ts's own code. Same interface as Engine, but async.
 *
 * float32 throughout on the GPU (sums in a different order from XLA and from engine.ts), so a real value within a
 * rounding error of a threshold can come out on the other side; engine.ts is the parity reference and /parity counts
 * any difference.
 */
import { Engine, embed, leadCompare, pnLayout, regPairs, startHops, startReg, type Inputs, type ModelConfig, type Params, type RegOut, type State, type Trace } from './engine';

const WG = 16;

const MATMUL = /* wgsl */ `
struct Dims { M: u32, K: u32, N: u32, ldw: u32, col0: u32, bias: u32, xoff: u32, ldx: u32 }
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> d: Dims;
var<workgroup> As: array<array<f32, ${WG}>, ${WG}>;
var<workgroup> Bs: array<array<f32, ${WG}>, ${WG}>;
@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
	let row = wg.y * ${WG}u + l.y;
	let col = wg.x * ${WG}u + l.x;
	var acc = 0.0;
	for (var t = 0u; t < d.K; t += ${WG}u) {
		let xr = min(row, d.M - 1u);
		let xk = min(t + l.x, d.K - 1u);
		let a = X[xr * d.ldx + d.xoff + xk];
		As[l.y][l.x] = select(0.0, a, row < d.M && t + l.x < d.K);
		let wk = min(t + l.y, d.K - 1u);
		let wc = min(col, d.N - 1u);
		let b = W[wk * d.ldw + d.col0 + wc];
		Bs[l.y][l.x] = select(0.0, b, t + l.y < d.K && col < d.N);
		workgroupBarrier();
		for (var k = 0u; k < ${WG}u; k++) { acc += As[l.y][k] * Bs[k][l.x]; }
		workgroupBarrier();
	}
	if (row < d.M && col < d.N) {
		var v = acc;
		if (d.bias != 0u) { v += B[d.col0 + col]; }
		Y[row * d.N + col] = v;
	}
}`;

const NORM = /* wgsl */ `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<uniform> p: vec4<u32>;
var<workgroup> s: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
	let r = wg.x + wg.y * 256u;
	if (r >= p.x) { return; }
	let D = p.y;
	let x = X[r * D + l.x];
	s[l.x] = x;
	workgroupBarrier();
	for (var w = 64u; w > 0u; w = w / 2u) { if (l.x < w) { s[l.x] += s[l.x + w]; } workgroupBarrier(); }
	let mean = s[0] / f32(D);
	workgroupBarrier();
	let dv = x - mean;
	s[l.x] = dv * dv;
	workgroupBarrier();
	for (var w = 64u; w > 0u; w = w / 2u) { if (l.x < w) { s[l.x] += s[l.x + w]; } workgroupBarrier(); }
	let v = s[0] / f32(D);
	Y[r * D + l.x] = dv * inverseSqrt(v + 1e-6);
}`;

/** Elementwise: mode 0 gelu(A) -> O; 1 A + B -> O; 2 A + sigmoid(G) * B -> O (G read at stride gs, offset go). */
const EW = /* wgsl */ `
struct P { len: u32, mode: u32, D: u32, gs: u32, go: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> Bv: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
	let i = g.x + g.y * 65535u * 256u;
	if (i >= p.len) { return; }
	let a = A[i];
	if (p.mode == 0u) {
		// tanh's argument clamped: Metal's tanh overflows to NaN for large inputs; |u| >= 15 is +-1 in float32 anyway
		let u = clamp(0.7978845608028654 * (a + 0.044715 * a * a * a), -15.0, 15.0);
		O[i] = 0.5 * a * (1.0 + tanh(u));
	} else if (p.mode == 1u) {
		O[i] = a + Bv[i];
	} else {
		let r = i / p.D;
		let c = i % p.D;
		let gv = G[r * p.gs + p.go + c];
		O[i] = a + (1.0 / (1.0 + exp(-gv))) * Bv[i];
	}
}`;

/** Count half (census): per pair (u, v) and channel pair q: N = #k with XA(u,k,c) > 0 and XC(k,v,c) > 0; bits more, same. */
const COUNT = /* wgsl */ `
struct P { n: u32, h: u32, S: u32, xc: u32, D: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<u32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
	let v = g.x; let u = g.y; let q = g.z;
	let n = p.n;
	if (u >= n || v >= n) { return; }
	let c1 = 2u * q; let c2 = c1 + 1u;
	var n1 = 0u; var n2 = 0u;
	for (var k = 0u; k < n; k++) {
		let ia = (u * n + k) * p.S;
		let ic = (k * n + v) * p.S + p.xc;
		if (X[ia + c1] > 0.0 && X[ic + c1] > 0.0) { n1++; }
		if (X[ia + c2] > 0.0 && X[ic + c2] > 0.0) { n2++; }
	}
	let i = u * n + v;
	Y[i * p.D + c1] = select(0.0, 1.0, n1 > n2);
	Y[i * p.D + c2] = select(0.0, 1.0, n1 == n2);
	C[i * p.h + c1] = n1;
	C[i * p.h + c2] = n2;
}`;

/** Best half: per (u, v, c): min over k of max(a, b) + t min(a, b), a, b = softplus(-score); value exp(-min) or the bit
 * [some k passes both tests]; and the best k. Scores at columns aoff + c (left) and xc + aoff + c (right), stride S. */
const BEST = /* wgsl */ `
struct P { n: u32, h: u32, S: u32, xc: u32, D: u32, aoff: u32, bits: u32, yoff: u32 }
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> T: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<storage, read_write> A: array<u32>;
@group(0) @binding(4) var<uniform> p: P;
fn sp(x: f32) -> f32 { return max(x, 0.0) + log(1.0 + exp(-abs(x))); }
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
	let v = g.x; let u = g.y; let c = g.z;
	let n = p.n;
	if (u >= n || v >= n) { return; }
	var best = 3.4e38; var arg = 255u; var any = false;
	let t = T[c];
	for (var k = 0u; k < n; k++) {
		let xa = X[(u * n + k) * p.S + p.aoff + c];
		let xb = X[(k * n + v) * p.S + p.xc + p.aoff + c];
		let a = sp(-xa); let b = sp(-xb);
		let cost = max(a, b) + t * min(a, b);
		if (cost < best) { best = cost; arg = k; }
		if (xa > 0.0 && xb > 0.0) { any = true; }
	}
	let i = u * n + v;
	Y[i * p.D + p.yoff + c] = select(exp(-best), select(0.0, 1.0, any), p.bits != 0u);
	A[i * p.h + c] = arg;
}`;

/** Pair-pointer read-out: logits[i, j, k] = sum_c Pm(i,k,c) Rm(k,j,c) / sqrt(D). */
const EDGE = /* wgsl */ `
struct P { n: u32, D: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<storage, read> Pm: array<f32>;
@group(0) @binding(1) var<storage, read> Rm: array<f32>;
@group(0) @binding(2) var<storage, read_write> L: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
	let k = g.x; let j = g.y; let i = g.z;
	let n = p.n;
	if (i >= n || j >= n || k >= n) { return; }
	var s = 0.0;
	for (var c = 0u; c < p.D; c++) { s += Pm[(i * n + k) * p.D + c] * Rm[(k * n + j) * p.D + c]; }
	L[(i * n + j) * n + k] = s / sqrt(f32(p.D));
}`;

type Buf = GPUBuffer;

export async function gpuAvailable(): Promise<GPUDevice | null> {
	const g = (globalThis.navigator as Navigator & { gpu?: GPU }).gpu;
	if (!g) return null;
	const adapter = await g.requestAdapter({ powerPreference: 'high-performance' });
	if (!adapter) return null;
	return adapter.requestDevice();
}

export class GpuEngine {
	readonly cpu: Engine;
	readonly D: number;
	readonly h: number;
	readonly M: number;
	readonly block: Engine['block'];
	private pipes: Record<string, GPUComputePipeline> = {};
	private w: Record<string, Buf> = {};
	private b: Record<string, Buf> = {};
	private n = 0;
	private x: Inputs | null = null;

	constructor(
		readonly dev: GPUDevice,
		readonly p: Params,
		readonly config: ModelConfig
	) {
		this.cpu = new Engine(p, config);
		this.D = this.cpu.D;
		this.h = this.cpu.h;
		this.block = this.cpu.block;
		this.M = p['blk/f1'].shape[1];
		if (this.D !== 128) throw new Error('the GPU norm is written for width 128');
		for (const [k, src] of Object.entries({ mm: MATMUL, norm: NORM, ew: EW, count: COUNT, best: BEST, edge: EDGE }))
			this.pipes[k] = dev.createComputePipeline({ layout: 'auto', compute: { module: dev.createShaderModule({ code: src }), entryPoint: 'main' } });
		const D = this.D;
		const cat = (names: string[], rows: number) => {
			const ts = names.map((nm) => p[nm]);
			const cols = ts.reduce((a, t) => a + (t.shape[1] ?? 1), 0);
			const out = new Float32Array(rows * cols);
			let off = 0;
			for (const t of ts) {
				const c = t.shape[1] ?? 1;
				for (let r = 0; r < rows; r++) for (let j = 0; j < c; j++) out[r * cols + off + j] = t.data[r * c + j];
				off += c;
			}
			return out;
		};
		const census = this.block === 'census';
		this.w.abg = this.upload(cat(census ? ['blk/a', 'blk/b', 'blk/g'] : ['blk/a', 'blk/b'], D));
		this.w.babg = this.upload(new Float32Array([...p['blk/ba'].data, ...p['blk/bb'].data, ...(census ? p['blk/bg'].data : [])]));
		for (const k of ['o', 'f1', 'f2']) {
			this.w[k] = this.upload(p[`blk/${k}`].data);
			this.w[`b${k}`] = this.upload(p[`blk/b${k}`].data);
		}
		this.w.theta = this.upload(this.cpu.theta);
		if (p['P']) {
			this.w.P = this.upload(p['P'].data);
			this.w.bP = this.upload(p['bP'].data);
			this.w.Rr = this.upload(p['Rr'].data);
			this.w.bR = this.upload(p['bR'].data);
		}
	}

	private upload(a: Float32Array): Buf {
		const buf = this.dev.createBuffer({ size: Math.max(16, a.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
		this.dev.queue.writeBuffer(buf, 0, a);
		return buf;
	}

	private buffer(bytes: number): Buf {
		return this.dev.createBuffer({ size: Math.max(16, bytes), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
	}

	private uniforms: Buf[] = [];
	private uniform(vals: number[]): Buf {
		const a = new Uint32Array(Math.max(4, Math.ceil(vals.length / 4) * 4));
		a.set(vals);
		const buf = this.dev.createBuffer({ size: a.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
		this.dev.queue.writeBuffer(buf, 0, a);
		this.uniforms.push(buf);
		return buf;
	}

	/** free the small parameter buffers of finished work */
	private async settle() {
		const done = this.uniforms;
		this.uniforms = [];
		await this.dev.queue.onSubmittedWorkDone();
		for (const b of done) b.destroy();
	}

	private run(enc: GPUCommandEncoder, pipe: string, bufs: Buf[], groups: [number, number, number]) {
		const pl = this.pipes[pipe];
		const bg = this.dev.createBindGroup({ layout: pl.getBindGroupLayout(0), entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })) });
		const pass = enc.beginComputePass();
		pass.setPipeline(pl);
		pass.setBindGroup(0, bg);
		pass.dispatchWorkgroups(...groups);
		pass.end();
	}

	private mm(enc: GPUCommandEncoder, X: Buf, rows: number, K: number, W: Buf, ldw: number, bias: Buf | null, Y: Buf, col0 = 0, N = ldw, xoff = 0, ldx = K) {
		this.run(enc, 'mm', [X, W, bias ?? W, Y, this.uniform([rows, K, N, ldw, col0, bias ? 1 : 0, xoff, ldx])], [Math.ceil(N / WG), Math.ceil(rows / WG), 1]);
	}

	private normRows(enc: GPUCommandEncoder, X: Buf, Y: Buf, rows: number) {
		this.run(enc, 'norm', [X, Y, this.uniform([rows, this.D, 0, 0])], [Math.min(rows, 256), Math.ceil(rows / 256), 1]);
	}

	private ew(enc: GPUCommandEncoder, mode: number, A: Buf, B: Buf, G: Buf, O: Buf, len: number, gs = 0, go = 0) {
		const groups = Math.ceil(len / 256);
		this.run(enc, 'ew', [A, B, G, O, this.uniform([len, mode, this.D, gs, go, 0, 0, 0])], [Math.min(groups, 65535), Math.ceil(groups / 65535), 1]);
	}

	private async download(buf: Buf, bytes: number): Promise<ArrayBuffer> {
		const st = this.dev.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
		const enc = this.dev.createCommandEncoder();
		enc.copyBufferToBuffer(buf, 0, st, 0, bytes);
		this.dev.queue.submit([enc.finish()]);
		await st.mapAsync(GPUMapMode.READ);
		const out = st.getMappedRange().slice(0);
		st.unmap();
		st.destroy();
		return out;
	}

	/** e + o(y) + MLP, from the bits y (resting blocks) into b.z. */
	private rebuild(enc: GPUCommandEncoder, y: Buf) {
		const { P } = this.dims();
		const { D, M } = this;
		this.mm(enc, y, P, D, this.w.o, D, this.w.bo, this.b.U);
		this.ew(enc, 1, this.b.e, this.b.U, this.b.U, this.b.s, P * D);
		this.mlp(enc, this.b.s, this.b.z);
		void M;
	}

	/** z_out = s + f2(gelu(f1(norm(s)))) */
	private mlp(enc: GPUCommandEncoder, s: Buf, out: Buf) {
		const { P } = this.dims();
		const { D, M } = this;
		this.normRows(enc, s, this.b.t, P);
		this.mm(enc, this.b.t, P, D, this.w.f1, M, this.w.bf1, this.b.H);
		this.ew(enc, 0, this.b.H, this.b.H, this.b.H, this.b.H2, P * M);
		this.mm(enc, this.b.H2, P, M, this.w.f2, D, this.w.bf2, this.b.U);
		this.ew(enc, 1, s, this.b.U, this.b.U, out, P * D);
	}

	private dims() {
		const n = this.n;
		return { n, P: n * n };
	}

	private allocate(n: number) {
		for (const b of Object.values(this.b)) b.destroy();
		this.b = {};
		this.n = n;
		const P = n * n;
		const { D, h, M } = this;
		const S = this.block === 'census' ? 3 * D : 2 * D;
		const f = 4;
		this.b.e = this.buffer(P * D * f);
		this.b.z = this.buffer(P * D * f);
		this.b.z2 = this.buffer(P * D * f);
		this.b.s = this.buffer(P * D * f);
		this.b.t = this.buffer(P * D * f);
		this.b.X = this.buffer(P * S * f);
		this.b.y = this.buffer(P * D * f);
		this.b.U = this.buffer(P * D * f);
		this.b.H = this.buffer(P * M * f);
		this.b.H2 = this.buffer(P * M * f);
		this.b.cnt = this.buffer(P * h * 4);
		this.b.arg = this.buffer(P * h * 4);
		if (this.w.P) {
			this.b.Pm = this.buffer(P * D * f);
			this.b.Rm = this.buffer(P * D * f);
			this.b.L = this.buffer(P * n * f);
		}
	}

	async start(x: Inputs): Promise<State> {
		this.x = x;
		this.cpu.x = x;
		this.allocate(x.n);
		const e = embed(this.p, x, this.D);
		this.dev.queue.writeBuffer(this.b.e, 0, e);
		if (this.block === 'census') {
			this.dev.queue.writeBuffer(this.b.z, 0, e);
			return { loop: 0, s: e, changed: e.length, trace: null };
		}
		// the first bits: zero, except the comparisons with fixed tests (the pair's own numbers)
		const y = new Float32Array(e.length);
		if (this.block === 'cold' || !this.cpu.kept) return { loop: 0, s: y, changed: y.length, trace: null };
		const { P } = this.dims();
		const R = this.block === 'pair_numbers' ? startReg(x, this.config.model.arithmetic!) : undefined;
		const H = this.cpu.v2 ? (this.config.model.table === 'hops' ? startHops(x, R!) : new Float32Array(P)) : undefined;
		const trace = this.newTrace();
		const c0 = new Float32Array(e.length);
		const built = await this.countHalf(y, c0, trace, R, H);
		const ownbit = this.ownbits();
		for (let i = 0; i < P; i++) for (let c = 0; c < this.h; c++) y[i * this.D + c] = c0[i * this.D + c] * ownbit[c];
		return { loop: 0, s: y, R, H, hits: this.cpu.v2 ? built!.hits : undefined, changed: y.length, trace: null };
	}

	private ownbits(): number[] {
		return this.cpu.ownbits();
	}

	private newTrace(): Trace {
		const { P } = this.dims();
		const h = this.h;
		return { testA: new Uint8Array(P * h), testB: new Uint8Array(P * h), counts: new Uint16Array(P * h), best: new Uint8Array(P * h), lead: this.block === 'census' ? undefined : new Float32Array(P * (h / 2)).fill(NaN) };
	}

	/** Resting blocks: rebuild the state from bits y, read the count half's test scores on the GPU, count the plain
	 * fact pairs there too (the first channel pairs: wire 0 and no fixed test), and compare the register pairs on the CPU. */
	private async countHalf(y: Float32Array, out: Float32Array, trace: Trace, R?: Float32Array, H?: Float32Array): Promise<RegOut | undefined> {
		const { n, P } = this.dims();
		const { D, h } = this;
		const S = D + h; // columns 0..h: left tests of the count half; D..D+h: its right tests
		const wireFull = this.block === 'compare_first' ? this.cpu.wire! : pnLayout(h).wire;
		let nf = 0;
		while (2 * nf + 1 < wireFull.length && wireFull[2 * nf] === 0 && wireFull[2 * nf + 1] === 0) nf++;
		this.dev.queue.writeBuffer(this.b.y, 0, y);
		const enc = this.dev.createCommandEncoder();
		this.rebuild(enc, this.b.y);
		this.normRows(enc, this.b.z, this.b.t, P);
		this.mm(enc, this.b.t, P, D, this.w.abg, 2 * D, this.w.babg, this.b.X, 0, S);
		if (nf) this.run(enc, 'count', [this.b.X, this.b.U, this.b.cnt, this.uniform([n, h, S, D, D, 0, 0, 0])], [Math.ceil(n / 8), Math.ceil(n / 8), nf]);
		this.dev.queue.submit([enc.finish()]);
		const [Xb, fy, fc] = await Promise.all([this.download(this.b.X, P * S * 4), this.download(this.b.U, P * D * 4), this.download(this.b.cnt, P * h * 4)]);
		const X = new Float32Array(Xb);
		const la = new Float32Array(P * D);
		const lc = new Float32Array(P * D);
		for (let i = 0; i < P; i++)
			for (let c = 0; c < h; c++) {
				la[i * D + c] = X[i * S + c];
				lc[i * D + c] = X[i * S + D + c];
			}
		const facts = nf ? { y: new Float32Array(fy), counts: new Uint32Array(fc) } : undefined;
		const x = this.x!;
		const C0 = performance.now();
		try {
			if (this.block === 'compare_first') {
				leadCompare(la, lc, n, h, D, x.reg!, this.cpu.wire!, this.cpu.own!, out, trace, facts);
				return undefined;
			}
			const { base, wire, own } = pnLayout(h, this.cpu.table);
			leadCompare(la, lc, n, h, D, x.reg!, wire, own, out, trace, facts);
			return regPairs(la, lc, n, h, D, base, x.W!, R!, this.config.model.arithmetic === 'plus', out, trace, this.cpu.table, H ?? null);
		} finally {
			this.lastCpu = performance.now() - C0;
		}
	}
	timing: Record<string, number> = {};
	private lastCpu = 0;

	async step(st: State): Promise<State> {
		const { n, P } = this.dims();
		const { D, h } = this;
		const trace = this.newTrace();
		if (this.block === 'census') {
			const S = 3 * D;
			const enc = this.dev.createCommandEncoder();
			this.normRows(enc, this.b.z, this.b.t, P);
			this.mm(enc, this.b.t, P, D, this.w.abg, S, this.w.babg, this.b.X);
			this.run(enc, 'count', [this.b.X, this.b.y, this.b.cnt, this.uniform([n, h, S, D, D, 0, 0, 0])], [Math.ceil(n / 8), Math.ceil(n / 8), h / 2]);
			this.run(enc, 'best', [this.b.X, this.w.theta, this.b.y, this.b.arg, this.uniform([n, h, S, D, D, h, 0, h])], [Math.ceil(n / 8), Math.ceil(n / 8), h]);
			this.mm(enc, this.b.y, P, D, this.w.o, D, this.w.bo, this.b.U);
			this.ew(enc, 2, this.b.z, this.b.U, this.b.X, this.b.z2, P * D, S, 2 * D);
			this.mlp(enc, this.b.z2, this.b.z);
			this.dev.queue.submit([enc.finish()]);
			const [z, X, cnt, arg] = await Promise.all([this.download(this.b.z, P * D * 4), this.download(this.b.X, P * S * 4), this.download(this.b.cnt, P * h * 4), this.download(this.b.arg, P * h * 4)]);
			const Xf = new Float32Array(X);
			for (let i = 0; i < P; i++)
				for (let c = 0; c < h; c++) {
					trace.testA[i * h + c] = Xf[i * S + c] > 0 ? 1 : 0;
					trace.testB[i * h + c] = Xf[i * S + D + c] > 0 ? 1 : 0;
				}
			trace.counts.set(new Uint32Array(cnt));
			trace.best.set(new Uint32Array(arg));
			const zf = new Float32Array(z);
			await this.settle();
			return { loop: st.loop + 1, s: zf, changed: zf.length, trace };
		}
		if (this.block === 'cold') {
			// both halves on the GPU, from the state rebuilt from the bits: more / same into columns 0..h, the exists bits h..D
			const y = st.s;
			this.dev.queue.writeBuffer(this.b.y, 0, y);
			const enc = this.dev.createCommandEncoder();
			this.rebuild(enc, this.b.y);
			this.normRows(enc, this.b.z, this.b.t, P);
			this.mm(enc, this.b.t, P, D, this.w.abg, 2 * D, this.w.babg, this.b.X, 0, 2 * D);
			this.run(enc, 'count', [this.b.X, this.b.U, this.b.cnt, this.uniform([n, h, 2 * D, D, D, 0, 0, 0])], [Math.ceil(n / 8), Math.ceil(n / 8), h / 2]);
			this.run(enc, 'best', [this.b.X, this.w.theta, this.b.U, this.b.arg, this.uniform([n, h, 2 * D, D, D, h, 1, h])], [Math.ceil(n / 8), Math.ceil(n / 8), h]);
			this.dev.queue.submit([enc.finish()]);
			const [yb, X, cnt, arg] = await Promise.all([this.download(this.b.U, P * D * 4), this.download(this.b.X, P * 2 * D * 4), this.download(this.b.cnt, P * h * 4), this.download(this.b.arg, P * h * 4)]);
			const y2 = new Float32Array(yb);
			const Xf = new Float32Array(X);
			for (let i = 0; i < P; i++)
				for (let c = 0; c < h; c++) {
					trace.testA[i * h + c] = Xf[i * 2 * D + c] > 0 ? 1 : 0;
					trace.testB[i * h + c] = Xf[i * 2 * D + D + c] > 0 ? 1 : 0;
				}
			trace.counts.set(new Uint32Array(cnt));
			trace.best.set(new Uint32Array(arg));
			let changed = 0;
			for (let i = 0; i < y2.length; i++) if (y2[i] !== y[i]) changed++;
			await this.settle();
			return { loop: st.loop + 1, s: y2, changed, trace };
		}
		// resting blocks: kept bits (exists half, on the GPU), then the count half (registers, on the CPU)
		const T0 = performance.now();
		const y = st.s;
		this.dev.queue.writeBuffer(this.b.y, 0, y);
		const enc = this.dev.createCommandEncoder();
		this.rebuild(enc, this.b.y);
		this.normRows(enc, this.b.z, this.b.t, P);
		this.mm(enc, this.b.t, P, D, this.w.abg, 2 * D, this.w.babg, this.b.X, 0, 2 * D);
		// exists bits into b.U (columns h..D), best witness into arg; left scores at columns h.., right at D + h..
		this.run(enc, 'best', [this.b.X, this.w.theta, this.b.U, this.b.arg, this.uniform([n, h, 2 * D, D, D, h, 1, h])], [Math.ceil(n / 8), Math.ceil(n / 8), h]);
		this.dev.queue.submit([enc.finish()]);
		const [ex, arg] = await Promise.all([this.download(this.b.U, P * D * 4), this.download(this.b.arg, P * h * 4)]);
		const exf = new Float32Array(ex);
		trace.best.set(new Uint32Array(arg));
		const ownbit = this.ownbits();
		const y2 = new Float32Array(P * D);
		if (!this.cpu.kept) {
			// compare_first without kept bits: the exists bits as read, and the count half from the same state (rebuilt from y)
			for (let i = 0; i < P; i++) for (let c = h; c < D; c++) y2[i * D + c] = exf[i * D + c];
			const cb = new Float32Array(P * D);
			await this.countHalf(y, cb, trace);
			for (let i = 0; i < P; i++) for (let c = 0; c < h; c++) y2[i * D + c] = cb[i * D + c];
			let changed = 0;
			for (let i = 0; i < y2.length; i++) if (y2[i] !== y[i]) changed++;
			await this.settle();
			return { loop: st.loop + 1, s: y2, changed, trace };
		}
		for (let i = 0; i < P; i++) {
			for (let c = 0; c < h; c++) y2[i * D + c] = y[i * D + c] * ownbit[c];
			for (let c = h; c < D; c++) y2[i * D + c] = Math.max(y[i * D + c], exf[i * D + c]);
		}
		const T1 = performance.now();
		const cb = new Float32Array(P * D);
		const built = await this.countHalf(y2, cb, trace, st.R, st.H);
		const T2 = performance.now();
		this.timing = { exists: T1 - T0, count: T2 - T1, countCpu: this.lastCpu };
		for (let i = 0; i < P; i++) for (let c = 0; c < h; c++) y2[i * D + c] = cb[i * D + c];
		await this.settle();
		return this.cpu.finish(st, y2, built, trace);
	}

	/** The read-outs: the state (rebuilt on the GPU for the resting blocks), the pair-pointer maps on the GPU. */
	async read(st: State) {
		const { n, P } = this.dims();
		const D = this.D;
		let z: Float32Array;
		const enc = this.dev.createCommandEncoder();
		if (this.block === 'census') {
			z = st.s;
		} else {
			this.dev.queue.writeBuffer(this.b.y, 0, st.s);
			this.rebuild(enc, this.b.y);
		}
		let edge: { Pm: Float32Array; Rm: Float32Array } | undefined;
		if (this.w.P) {
			if (this.block === 'census') this.dev.queue.writeBuffer(this.b.z, 0, st.s);
			this.normRows(enc, this.b.z, this.b.t, P);
			this.mm(enc, this.b.t, P, D, this.w.P, D, this.w.bP, this.b.Pm);
			this.mm(enc, this.b.t, P, D, this.w.Rr, D, this.w.bR, this.b.Rm);
		}
		this.dev.queue.submit([enc.finish()]);
		if (this.block !== 'census') z = new Float32Array(await this.download(this.b.z, P * D * 4));
		if (this.w.P) {
			const [Pm, Rm] = await Promise.all([this.download(this.b.Pm, P * D * 4), this.download(this.b.Rm, P * D * 4)]);
			edge = { Pm: new Float32Array(Pm), Rm: new Float32Array(Rm) };
		}
		await this.settle();
		return this.cpu.readState(z!, n, edge, st.hits);
	}

	destroy() {
		for (const b of [...Object.values(this.b), ...Object.values(this.w)]) b.destroy();
		this.b = {};
		this.w = {};
	}
}

