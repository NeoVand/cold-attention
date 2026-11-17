/**
 * Reference answers for the 30 CLRS-30 tasks: ports of dm-clrs 2.0.3's clrs/_src/algorithms/*.py without probes or
 * hints, with the library's loop orders, comparisons and tie-breaking, in float64 as numpy computes them. solve()
 * returns the outputs exactly as a split file stores them (see instances.ts): pointers and indices as integers,
 * masks as 0/1 (bridges marks non-edges -1), lcs_length's arrows as class indices with -1 where the library masks a cell.
 *
 * Graph tasks read the edge weights A as stored (float32) and compute in float64; on the official splits this gives the
 * library's outputs on its float64 weights for every instance.
 *
 * Where the library calls numpy's default argsort (introsort, not stable: Kruskal's edge order, activity_selector,
 * task_scheduling, the SCC second pass, graham_scan's angle order, Prim's and Dijkstra's extract-min), exact ties
 * are broken here by the lower index. numpy's choice on an exact tie depends on the array length and the CPU's sort
 * kernel; continuous random inputs have no ties.
 */
import { isGraph, type Instance, type Task, type TypedArray, type Value } from './instances';

export type Outputs = Record<string, Value>;

const f64 = (v: Value | undefined, name: string): Float64Array => {
	if (v === undefined || typeof v === 'number') throw new Error(`input ${name} missing or not an array`);
	return Float64Array.from(v as ArrayLike<number>);
};
const scalar = (v: Value | undefined, name: string): number => {
	if (typeof v !== 'number') throw new Error(`input ${name} missing or not a number`);
	return v;
};
const arange = (n: number): Int32Array => {
	const a = new Int32Array(n);
	for (let i = 0; i < n; i++) a[i] = i;
	return a;
};
const i16 = (a: ArrayLike<number>): Int16Array => Int16Array.from(a);

/** Indices that sort `keys` ascending; exact ties by the lower index. */
function argsort(keys: ArrayLike<number>): number[] {
	return Array.from({ length: keys.length }, (_, i) => i).sort((a, b) => keys[a] - keys[b] || a - b);
}

/** numpy's argsort(x)[0]: the first index of the minimum (see the module note on ties). */
function argmin(x: ArrayLike<number>): number {
	let best = 0;
	for (let i = 1; i < x.length; i++) if (x[i] < x[best]) best = i;
	return best;
}

/** probing.array: the predecessor of each item in the order A_pos (the first item points to itself). */
function predArray(A_pos: ArrayLike<number>): Int16Array {
	const n = A_pos.length;
	const probe = i16(arange(n));
	for (let i = 1; i < n; i++) probe[A_pos[i]] = A_pos[i - 1];
	return probe;
}

// ---------------------------------------------------------------------------------------------------------------
// Sorting

function insertionSort(key: Float64Array): Outputs {
	const A = key.slice();
	const n = A.length;
	const A_pos = arange(n);
	for (let j = 1; j < n; j++) {
		const k = A[j];
		let i = j - 1;
		while (i >= 0 && A[i] > k) {
			A[i + 1] = A[i];
			A_pos[i + 1] = A_pos[i];
			i -= 1;
		}
		A[i + 1] = k;
		A_pos[i + 1] = j;
	}
	return { pred: predArray(A_pos) };
}

function bubbleSort(key: Float64Array): Outputs {
	const A = key.slice();
	const n = A.length;
	const A_pos = arange(n);
	for (let i = 0; i < n - 1; i++) {
		for (let j = n - 1; j >= i + 1; j--) {
			if (A[j] < A[j - 1]) {
				[A[j], A[j - 1]] = [A[j - 1], A[j]];
				[A_pos[j], A_pos[j - 1]] = [A_pos[j - 1], A_pos[j]];
			}
		}
	}
	return { pred: predArray(A_pos) };
}

function heapsort(key: Float64Array): Outputs {
	const A = key.slice();
	const n = A.length;
	const A_pos = arange(n);
	const swap = (i: number, j: number) => {
		[A[i], A[j]] = [A[j], A[i]];
		[A_pos[i], A_pos[j]] = [A_pos[j], A_pos[i]];
	};
	const maxHeapify = (start: number, heapSize: number) => {
		let i = start;
		for (;;) {
			const l = 2 * i + 1;
			const r = 2 * i + 2;
			let largest = l < heapSize && A[l] > A[i] ? l : i;
			if (r < heapSize && A[r] > A[largest]) largest = r;
			if (largest === i) return;
			swap(i, largest);
			i = largest;
		}
	};
	for (let i = n - 1; i >= 0; i--) maxHeapify(i, n);
	let heapSize = n;
	for (let i = n - 1; i >= 1; i--) {
		swap(0, i);
		heapSize -= 1;
		maxHeapify(0, heapSize);
	}
	return { pred: predArray(A_pos) };
}

/** The Lomuto partition of the library's quicksort and quickselect: pivot A[r], `A[j] <= x` goes left. */
function partition(A: Float64Array, A_pos: Int32Array, p: number, r: number): number {
	const x = A[r];
	let i = p - 1;
	for (let j = p; j < r; j++) {
		if (A[j] <= x) {
			i += 1;
			[A[i], A[j]] = [A[j], A[i]];
			[A_pos[i], A_pos[j]] = [A_pos[j], A_pos[i]];
		}
	}
	[A[i + 1], A[r]] = [A[r], A[i + 1]];
	[A_pos[i + 1], A_pos[r]] = [A_pos[r], A_pos[i + 1]];
	return i + 1;
}

function quicksort(key: Float64Array): Outputs {
	const A = key.slice();
	const A_pos = arange(A.length);
	const stack: [number, number][] = [[0, A.length - 1]];
	while (stack.length) {
		const [p, r] = stack.pop()!;
		if (p < r) {
			const q = partition(A, A_pos, p, r);
			// The library recurses on the left part first; the order does not change the result (disjoint ranges).
			stack.push([q + 1, r], [p, q - 1]);
		}
	}
	return { pred: predArray(A_pos) };
}

// ---------------------------------------------------------------------------------------------------------------
// Searching

function minimum(key: Float64Array): Outputs {
	let min = 0;
	for (let i = 1; i < key.length; i++) if (key[min] > key[i]) min = i;
	return { min };
}

function binarySearch(key: Float64Array, x: number): Outputs {
	let low = 0;
	let high = key.length - 1;
	while (low < high) {
		const mid = Math.floor((low + high) / 2);
		if (x <= key[mid]) high = mid;
		else low = mid + 1;
	}
	return { return: high };
}

/** Quickselect for rank floor(n / 2) (the upper median), as the library calls it. */
function quickselect(key: Float64Array): Outputs {
	const A = key.slice();
	const A_pos = arange(A.length);
	let p = 0;
	let r = A.length - 1;
	let i = Math.floor(A.length / 2);
	for (;;) {
		const q = partition(A, A_pos, p, r);
		const k = q - p;
		if (i === k) return { median: A_pos[q] };
		if (i < k) r = q - 1;
		else {
			p = q + 1;
			i = i - k - 1;
		}
	}
}

// ---------------------------------------------------------------------------------------------------------------
// Divide and conquer, dynamic programming

function kadane(A: Float64Array): Outputs {
	let bestLow = 0;
	let bestHigh = 0;
	let bestSum = A[0];
	let i = 0;
	let sum = A[0];
	for (let j = 1; j < A.length; j++) {
		const x = A[j];
		if (sum + x >= x) sum += x;
		else {
			i = j;
			sum = x;
		}
		if (sum > bestSum) {
			bestLow = i;
			bestHigh = j;
			bestSum = sum;
		}
	}
	return { start: bestLow, end: bestHigh };
}

/** The library's matrix_chain_order: rounds of relaxation over all (i, j, k) from the previous round's table. */
function matrixChainOrder(p: Float64Array): Outputs {
	const n = p.length;
	const m = new Float64Array(n * n);
	const s = new Float64Array(n * n);
	const msk = new Uint8Array(n * n);
	for (let i = 1; i < n; i++) msk[i * n + i] = 1;
	for (;;) {
		const prevM = m.slice();
		const prevMsk = msk.slice();
		for (let i = 1; i < n; i++) {
			for (let j = i + 1; j < n; j++) {
				let flag = prevMsk[i * n + j];
				for (let k = i; k < j; k++) {
					if (prevMsk[i * n + k] === 1 && prevMsk[(k + 1) * n + j] === 1) {
						msk[i * n + j] = 1;
						const q = prevM[i * n + k] + prevM[(k + 1) * n + j] + p[i - 1] * p[k] * p[j];
						if (flag === 0 || q < m[i * n + j]) {
							m[i * n + j] = q;
							s[i * n + j] = k;
							flag = 1;
						}
					}
				}
			}
		}
		let same = true;
		for (let t = 0; t < n * n && same; t++) if (prevM[t] !== m[t]) same = false;
		if (same) break;
	}
	return { s: i16(s) };
}

/**
 * lcs_length over the two strings x (string = 0) and y (string = 1). The library iterates rounds until the table c
 * stops changing and then reads b from the converged table; the fixed point is unique, so one pass in order gives the
 * same b. Output: the (|x|+|y|)^2 table of classes, b[i, j] at row i, column |x| + j; every other cell -1 (masked).
 */
function lcsLength(x: Int32Array, y: Int32Array): Outputs {
	const nx = x.length;
	const ny = y.length;
	const c = new Float64Array(nx * ny);
	const b = new Int32Array(nx * ny);
	for (let i = 0; i < nx; i++) {
		if (x[i] === y[0]) {
			c[i * ny] = 1;
			b[i * ny] = 0;
		} else if (i > 0 && c[(i - 1) * ny] === 1) {
			c[i * ny] = 1;
			b[i * ny] = 1;
		} else {
			c[i * ny] = 0;
			b[i * ny] = 1;
		}
	}
	for (let j = 0; j < ny; j++) {
		if (x[0] === y[j]) {
			c[j] = 1;
			b[j] = 0;
		} else if (j > 0 && c[j - 1] === 1) {
			c[j] = 1;
			b[j] = 2;
		} else {
			c[j] = 0;
			b[j] = 1;
		}
	}
	for (let i = 1; i < nx; i++) {
		for (let j = 1; j < ny; j++) {
			if (x[i] === y[j]) {
				c[i * ny + j] = c[(i - 1) * ny + j - 1] + 1;
				b[i * ny + j] = 0;
			} else if (c[(i - 1) * ny + j] >= c[i * ny + j - 1]) {
				c[i * ny + j] = c[(i - 1) * ny + j];
				b[i * ny + j] = 1;
			} else {
				c[i * ny + j] = c[i * ny + j - 1];
				b[i * ny + j] = 2;
			}
		}
	}
	const N = nx + ny;
	const out = new Int16Array(N * N).fill(-1);
	for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) out[i * N + nx + j] = b[i * ny + j];
	return { b: out };
}

/** optimal_bst over n = L + 1 items: p (L key probabilities, then a 0) and q (L + 1 gap probabilities). */
function optimalBst(pIn: Float64Array, q: Float64Array): Outputs {
	const N = q.length;
	const L = N - 1;
	const p = pIn.subarray(0, L);
	const e = new Float64Array(N * N);
	const w = new Float64Array(N * N);
	const root = new Int16Array(N * N);
	for (let i = 0; i < N; i++) {
		e[i * N + i] = q[i];
		w[i * N + i] = q[i];
	}
	for (let l = 1; l <= L; l++) {
		for (let i = 0; i < L - l + 1; i++) {
			const j = i + l;
			e[i * N + j] = 1e9;
			w[i * N + j] = w[i * N + j - 1] + p[j - 1] + q[j];
			for (let r = i; r < j; r++) {
				const t = e[i * N + r] + e[(r + 1) * N + j] + w[i * N + j];
				if (t < e[i * N + j]) {
					e[i * N + j] = t;
					root[i * N + j] = r;
				}
			}
		}
	}
	return { root };
}

// ---------------------------------------------------------------------------------------------------------------
// Greedy

function activitySelector(s: Float64Array, f: Float64Array): Outputs {
	const n = s.length;
	const A = new Int16Array(n);
	const ind = argsort(f);
	A[ind[0]] = 1;
	let k = ind[0];
	for (let m = 1; m < n; m++) {
		if (s[ind[m]] >= f[k]) {
			A[ind[m]] = 1;
			k = ind[m];
		}
	}
	return { selected: A };
}

function taskScheduling(d: Float64Array, w: Float64Array): Outputs {
	const n = d.length;
	const A = new Int16Array(n);
	const ind = argsort(w.map((v) => -v));
	A[ind[0]] = 1;
	let t = 1;
	for (let i = 1; i < n; i++) {
		if (t < d[ind[i]]) {
			A[ind[i]] = 1;
			t += 1;
		}
	}
	return { selected: A };
}

// ---------------------------------------------------------------------------------------------------------------
// Strings: T (haystack, string = 0) then P (needle, string = 1) in one sequence; match = index of the first match's
// start in T, or |T| (the needle's head) if there is none.

function naiveStringMatcher(T: Int32Array, P: Int32Array): Outputs {
	for (let s = 0; s <= T.length - P.length; s++) {
		let i = s;
		let j = 0;
		for (;;) {
			if (T[i] !== P[j]) break;
			if (j === P.length - 1) return { match: s };
			i += 1;
			j += 1;
		}
	}
	return { match: T.length };
}

function kmpMatcher(T: Int32Array, P: Int32Array): Outputs {
	const m = P.length;
	const pi = arange(m);
	const isReset = new Uint8Array(m);
	let k = 0;
	let kReset = 1;
	isReset[0] = 1;
	for (let q = 1; q < m; q++) {
		while (kReset === 0 && P[k + 1] !== P[q]) {
			if (isReset[k] === 1) {
				kReset = 1;
				k = 0;
			} else k = pi[k];
		}
		if (kReset === 1) {
			kReset = 0;
			k = -1;
		}
		if (P[k + 1] === P[q]) k += 1;
		if (k === -1) {
			k = 0;
			kReset = 1;
			isReset[q] = 1;
		}
		pi[q] = k;
	}
	let q = 0;
	let qReset = 1;
	let s = 0;
	for (let i = 0; i < T.length; i++) {
		if (i >= m) s += 1;
		while (qReset === 0 && P[q + 1] !== T[i]) {
			if (isReset[q] === 1) {
				q = 0;
				qReset = 1;
			} else q = pi[q];
		}
		if (qReset === 1) {
			q = -1;
			qReset = 0;
		}
		if (P[q + 1] === T[i]) {
			if (q === m - 2) return { match: s };
			q += 1;
		}
		if (q === -1) {
			qReset = 1;
			q = 0;
		}
	}
	return { match: T.length };
}

// ---------------------------------------------------------------------------------------------------------------
// Geometry

function segmentsIntersect(xs: Float64Array, ys: Float64Array): Outputs {
	const cross = (x1: number, y1: number, x2: number, y2: number) => x1 * y2 - x2 * y1;
	const direction = (i: number, j: number, k: number) =>
		cross(xs[k] - xs[i], ys[k] - ys[i], xs[j] - xs[i], ys[j] - ys[i]);
	const onSegment = (i: number, j: number, k: number) =>
		Math.min(xs[i], xs[j]) <= xs[k] &&
		xs[k] <= Math.max(xs[i], xs[j]) &&
		Math.min(ys[i], ys[j]) <= ys[k] &&
		ys[k] <= Math.max(ys[i], ys[j]);
	const dirs = [direction(2, 3, 0), direction(2, 3, 1), direction(0, 1, 2), direction(0, 1, 3)];
	const onSeg = [onSegment(2, 3, 0), onSegment(2, 3, 1), onSegment(0, 1, 2), onSegment(0, 1, 3)];
	let ret = 0;
	if (
		((dirs[0] > 0 && dirs[1] < 0) || (dirs[0] < 0 && dirs[1] > 0)) &&
		((dirs[2] > 0 && dirs[3] < 0) || (dirs[2] < 0 && dirs[3] > 0))
	)
		ret = 1;
	else if (dirs.some((d, t) => d === 0 && onSeg[t])) ret = 1;
	return { intersect: ret };
}

/** The lowest point (then leftmost), as both hull algorithms pick it. */
function lowest(xs: Float64Array, ys: Float64Array): number {
	let best = 0;
	for (let i = 0; i < xs.length; i++) if (ys[i] < ys[best] || (ys[i] === ys[best] && xs[i] < xs[best])) best = i;
	return best;
}

function grahamScan(xs: Float64Array, ys: Float64Array): Outputs {
	const n = xs.length;
	const inHull = new Int16Array(n);
	const stackPrev = arange(n);
	const ccw = (i: number, j: number, k: number) =>
		(xs[j] - xs[i]) * (ys[k] - ys[i]) - (ys[j] - ys[i]) * (xs[k] - xs[i]) <= 0;
	const best = lowest(xs, ys);
	inHull[best] = 1;
	let lastStack = best;
	const atans = new Float64Array(n);
	for (let i = 0; i < n; i++) if (i !== best) atans[i] = Math.atan2(ys[i] - ys[best], xs[i] - xs[best]);
	atans[best] = -123456789;
	const ind = argsort(atans);
	for (let i = 1; i < n; i++) {
		if (i >= 3) {
			while (ccw(stackPrev[lastStack], lastStack, ind[i])) {
				const prevLast = lastStack;
				lastStack = stackPrev[lastStack];
				stackPrev[prevLast] = prevLast;
				inHull[prevLast] = 0;
			}
		}
		inHull[ind[i]] = 1;
		stackPrev[ind[i]] = lastStack;
		lastStack = ind[i];
	}
	return { in_hull: inHull };
}

function jarvisMarch(xs: Float64Array, ys: Float64Array): Outputs {
	const n = xs.length;
	const inHull = new Int16Array(n);
	const ccw = (i: number, j: number, k: number) => {
		if (k === i || k === j) return false;
		return (xs[j] - xs[i]) * (ys[k] - ys[i]) - (ys[j] - ys[i]) * (xs[k] - xs[i]) <= 0;
	};
	const best = lowest(xs, ys);
	inHull[best] = 1;
	let lastPoint = best;
	let endpoint = 0;
	for (;;) {
		for (let i = 0; i < n; i++) if (endpoint === lastPoint || ccw(lastPoint, endpoint, i)) endpoint = i;
		if (inHull[endpoint] > 0) break;
		inHull[endpoint] = 1;
		lastPoint = endpoint;
		endpoint = 0;
	}
	return { in_hull: inHull };
}

// ---------------------------------------------------------------------------------------------------------------
// Graphs: A is n*n row-major, A[u * n + v] the weight of u -> v (0 = no edge).

/**
 * The library's iterative DFS (dfs, articulation_points, bridges and both SCC passes share it): from each root in
 * `roots` not yet visited, descend to the lowest-numbered unvisited neighbour; times advance by 0.01 per discovery and
 * finish. Hooks see each visited-neighbour scan and each finish.
 */
interface DfsHooks {
	onDiscover?: (u: number, time: number) => void;
	onTree?: (u: number, v: number) => void;
	onBack?: (u: number, v: number) => void;
	onFinish?: (u: number, time: number) => void;
	onVisit?: (u: number, root: number) => void;
}

function iterativeDfs(A: Float64Array, n: number, roots: ArrayLike<number>, d: Float64Array, h: DfsHooks): number {
	const color = new Int32Array(n);
	const sPrev = arange(n);
	let time = 0;
	for (let r = 0; r < roots.length; r++) {
		const s = roots[r];
		if (color[s] !== 0) continue;
		let sLast = s;
		let u = s;
		for (;;) {
			h.onVisit?.(u, s);
			if (color[u] === 0 || d[u] === 0) {
				time += 0.01;
				d[u] = time;
				color[u] = 1;
				h.onDiscover?.(u, time);
			}
			for (let v = 0; v < n; v++) {
				if (A[u * n + v] !== 0) {
					if (color[v] === 0) {
						color[v] = 1;
						sPrev[v] = sLast;
						sLast = v;
						h.onTree?.(u, v);
						break;
					} else h.onBack?.(u, v);
				}
			}
			if (sLast === u) {
				color[u] = 2;
				time += 0.01;
				h.onFinish?.(u, time);
				if (sPrev[u] === u) break;
				const pr = sPrev[sLast];
				sPrev[sLast] = sLast;
				sLast = pr;
			}
			u = sLast;
		}
	}
	return time;
}

function dfs(A: Float64Array, n: number): Outputs {
	const pi = arange(n);
	iterativeDfs(A, n, arange(n), new Float64Array(n), { onTree: (u, v) => (pi[v] = u) });
	return { pi: i16(pi) };
}

function articulationPoints(A: Float64Array, n: number): Outputs {
	const pi = arange(n);
	const d = new Float64Array(n);
	const low = new Float64Array(n);
	const childCnt = new Float64Array(n);
	const isCut = new Int16Array(n);
	iterativeDfs(A, n, arange(n), d, {
		onDiscover: (u, time) => (low[u] = time),
		onTree: (u, v) => {
			pi[v] = u;
			childCnt[u] += 0.01;
		},
		onBack: (u, v) => {
			if (v !== pi[u]) low[u] = Math.min(low[u], d[v]);
		},
		onFinish: (u) => {
			for (let v = 0; v < n; v++) {
				if (pi[v] === u) {
					low[u] = Math.min(low[u], low[v]);
					if (pi[u] !== u && low[v] >= d[u]) isCut[u] = 1;
				}
			}
			if (pi[u] === u && childCnt[u] > 0.01) isCut[u] = 1;
		}
	});
	return { is_cut: isCut };
}

function bridges(A: Float64Array, n: number): Outputs {
	const pi = arange(n);
	const d = new Float64Array(n);
	const low = new Float64Array(n);
	const isBridge = new Int16Array(n * n);
	// -1 + probing.graph(A): 0 on edges and the diagonal, -1 elsewhere.
	for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) isBridge[i * n + j] = A[i * n + j] + (i === j ? 1 : 0) !== 0 ? 0 : -1;
	iterativeDfs(A, n, arange(n), d, {
		onDiscover: (u, time) => (low[u] = time),
		onTree: (u, v) => (pi[v] = u),
		onBack: (u, v) => {
			if (v !== pi[u]) low[u] = Math.min(low[u], d[v]);
		},
		onFinish: (u) => {
			for (let v = 0; v < n; v++) {
				if (pi[v] === u) {
					low[u] = Math.min(low[u], low[v]);
					if (low[v] > d[u]) {
						isBridge[u * n + v] = 1;
						isBridge[v * n + u] = 1;
					}
				}
			}
		}
	});
	return { is_bridge: isBridge };
}

function stronglyConnectedComponents(A: Float64Array, n: number): Outputs {
	const d = new Float64Array(n);
	const f = new Float64Array(n);
	iterativeDfs(A, n, arange(n), d, { onFinish: (u, time) => (f[u] = time) });
	const At = new Float64Array(n * n);
	for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) At[i * n + j] = A[j * n + i];
	const order = argsort(f.map((v) => -v));
	const sccId = arange(n);
	// The second pass keeps d from the first (only the roots' times change, which no output reads).
	iterativeDfs(At, n, order, d, { onVisit: (u, root) => (sccId[u] = root) });
	return { scc_id: i16(sccId) };
}

function topologicalSort(A: Float64Array, n: number): Outputs {
	const color = new Int32Array(n);
	const topo = arange(n);
	const sPrev = arange(n);
	let topoHead = 0;
	for (let s = 0; s < n; s++) {
		if (color[s] !== 0) continue;
		let sLast = s;
		let u = s;
		for (;;) {
			if (color[u] === 0) color[u] = 1;
			for (let v = 0; v < n; v++) {
				if (A[u * n + v] !== 0 && color[v] === 0) {
					color[v] = 1;
					sPrev[v] = sLast;
					sLast = v;
					break;
				}
			}
			if (sLast === u) {
				color[u] = 2;
				if (color[topoHead] === 2) topo[u] = topoHead;
				topoHead = u;
				if (sPrev[u] === u) break;
				const pr = sPrev[sLast];
				sPrev[sLast] = sLast;
				sLast = pr;
			}
			u = sLast;
		}
	}
	return { topo: i16(topo), topo_head: topoHead };
}

function bfs(A: Float64Array, n: number, s: number): Outputs {
	const reach = new Uint8Array(n);
	const pi = arange(n);
	reach[s] = 1;
	for (;;) {
		const prev = reach.slice();
		for (let i = 0; i < n; i++) {
			if (prev[i] !== 1) continue;
			for (let j = 0; j < n; j++) {
				if (A[i * n + j] > 0) {
					if (pi[j] === j && j !== s) pi[j] = i;
					reach[j] = 1;
				}
			}
		}
		let same = true;
		for (let i = 0; i < n; i++) if (reach[i] !== prev[i]) same = false;
		if (same) break;
	}
	return { pi: i16(pi) };
}

function mstKruskal(A: Float64Array, n: number): Outputs {
	const pi = arange(n);
	const inMst = new Int16Array(n * n);
	const lx: number[] = [];
	const ly: number[] = [];
	const wts: number[] = [];
	for (let i = 0; i < n; i++) {
		for (let j = i + 1; j < n; j++) {
			if (A[i * n + j] > 0) {
				lx.push(i);
				ly.push(j);
				wts.push(A[i * n + j]);
			}
		}
	}
	const findRoot = (x: number): number => {
		// The library's find: walk to the root, pointing every node met so far at the current ancestor.
		let root = x;
		const mask = new Uint8Array(n);
		mask[x] = 1;
		while (pi[root] !== root) {
			root = pi[root];
			for (let i = 0; i < n; i++) if (mask[i] === 1) pi[i] = root;
			mask[root] = 1;
		}
		return root;
	};
	for (const ind of argsort(wts)) {
		const u = lx[ind];
		const v = ly[ind];
		const rootU = findRoot(u);
		const rootV = findRoot(v);
		if (rootU < rootV) {
			inMst[u * n + v] = inMst[v * n + u] = 1;
			pi[rootU] = rootV;
		} else if (rootU > rootV) {
			inMst[u * n + v] = inMst[v * n + u] = 1;
			pi[rootV] = rootU;
		}
	}
	return { in_mst: inMst };
}

function mstPrim(A: Float64Array, n: number, s: number): Outputs {
	const key = new Float64Array(n);
	const mark = new Uint8Array(n);
	const inQueue = new Float64Array(n);
	const pi = arange(n);
	inQueue[s] = 1;
	const score = new Float64Array(n);
	for (let it = 0; it < n; it++) {
		for (let i = 0; i < n; i++) score[i] = key[i] + (1.0 - inQueue[i]) * 1e9;
		const u = argmin(score);
		if (inQueue[u] === 0) break;
		mark[u] = 1;
		inQueue[u] = 0;
		for (let v = 0; v < n; v++) {
			const a = A[u * n + v];
			if (a !== 0 && mark[v] === 0 && (inQueue[v] === 0 || a < key[v])) {
				pi[v] = u;
				key[v] = a;
				inQueue[v] = 1;
			}
		}
	}
	return { pi: i16(pi) };
}

function bellmanFord(A: Float64Array, n: number, s: number): Outputs {
	const d = new Float64Array(n);
	const pi = arange(n);
	const msk = new Uint8Array(n);
	msk[s] = 1;
	for (;;) {
		const prevD = d.slice();
		const prevMsk = msk.slice();
		for (let u = 0; u < n; u++) {
			if (prevMsk[u] !== 1) continue;
			for (let v = 0; v < n; v++) {
				const a = A[u * n + v];
				if (a !== 0) {
					if (msk[v] === 0 || prevD[u] + a < d[v]) {
						d[v] = prevD[u] + a;
						pi[v] = u;
					}
					msk[v] = 1;
				}
			}
		}
		let same = true;
		for (let i = 0; i < n; i++) if (d[i] !== prevD[i]) same = false;
		if (same) break;
	}
	return { pi: i16(pi) };
}

function dijkstra(A: Float64Array, n: number, s: number): Outputs {
	const d = new Float64Array(n);
	const mark = new Uint8Array(n);
	const inQueue = new Float64Array(n);
	const pi = arange(n);
	inQueue[s] = 1;
	const score = new Float64Array(n);
	for (let it = 0; it < n; it++) {
		for (let i = 0; i < n; i++) score[i] = d[i] + (1.0 - inQueue[i]) * 1e9;
		const u = argmin(score);
		if (inQueue[u] === 0) break;
		mark[u] = 1;
		inQueue[u] = 0;
		for (let v = 0; v < n; v++) {
			const a = A[u * n + v];
			if (a !== 0 && mark[v] === 0 && (inQueue[v] === 0 || d[u] + a < d[v])) {
				pi[v] = u;
				d[v] = d[u] + a;
				inQueue[v] = 1;
			}
		}
	}
	return { pi: i16(pi) };
}

function dagShortestPaths(A: Float64Array, n: number, s: number): Outputs {
	const pi = arange(n);
	const d = new Float64Array(n);
	const mark = new Uint8Array(n);
	const color = new Int32Array(n);
	const topo = arange(n);
	const sPrev = arange(n);
	let topoHead = 0;
	let sLast = s;
	let u = s;
	// Topological order of the nodes reachable from s (one DFS from s).
	for (;;) {
		if (color[u] === 0) color[u] = 1;
		for (let v = 0; v < n; v++) {
			if (A[u * n + v] !== 0 && color[v] === 0) {
				color[v] = 1;
				sPrev[v] = sLast;
				sLast = v;
				break;
			}
		}
		if (sLast === u) {
			color[u] = 2;
			if (color[topoHead] === 2) topo[u] = topoHead;
			topoHead = u;
			if (sPrev[u] === u) break;
			const pr = sPrev[sLast];
			sPrev[sLast] = sLast;
			sLast = pr;
		}
		u = sLast;
	}
	if (topoHead !== s) throw new Error('dag_shortest_paths: the DFS did not end at the source');
	d[topoHead] = 0;
	mark[topoHead] = 1;
	while (topo[topoHead] !== topoHead) {
		const i = topoHead;
		mark[topoHead] = 1;
		for (let j = 0; j < n; j++) {
			const a = A[i * n + j];
			if (a !== 0 && (mark[j] === 0 || d[i] + a < d[j])) {
				d[j] = d[i] + a;
				pi[j] = i;
				mark[j] = 1;
			}
		}
		topoHead = topo[topoHead];
	}
	return { pi: i16(pi) };
}

function floydWarshall(A: Float64Array, n: number): Outputs {
	const D = A.slice();
	const Pi = new Int32Array(n * n);
	const msk = new Uint8Array(n * n);
	for (let i = 0; i < n; i++) {
		for (let j = 0; j < n; j++) {
			Pi[i * n + j] = i;
			msk[i * n + j] = A[i * n + j] + (i === j ? 1 : 0) !== 0 ? 1 : 0;
		}
	}
	for (let k = 0; k < n; k++) {
		const prevD = D.slice();
		const prevMsk = msk.slice();
		for (let i = 0; i < n; i++) {
			if (!(prevMsk[i * n + k] > 0)) continue;
			for (let j = 0; j < n; j++) {
				if (prevMsk[k * n + j] > 0) {
					const t = prevD[i * n + k] + prevD[k * n + j];
					if (msk[i * n + j] === 0 || t < D[i * n + j]) {
						D[i * n + j] = t;
						Pi[i * n + j] = Pi[k * n + j];
					} else D[i * n + j] = prevD[i * n + j];
					msk[i * n + j] = 1;
				}
			}
		}
	}
	return { Pi: i16(Pi) };
}

// ---------------------------------------------------------------------------------------------------------------

/** Characters of the two strings of a string task (key one-hot n*4, string 0 = first string / haystack). */
function twoStrings(inst: Instance): [Int32Array, Int32Array] {
	const key = inst.inputs.key as TypedArray;
	const str = inst.inputs.string as TypedArray;
	const C = key.length / inst.n;
	const a: number[] = [];
	const b: number[] = [];
	for (let i = 0; i < inst.n; i++) {
		let c = 0;
		for (let t = 1; t < C; t++) if (key[i * C + t] > key[i * C + c]) c = t;
		(str[i] === 0 ? a : b).push(c);
	}
	return [Int32Array.from(a), Int32Array.from(b)];
}

/** The true outputs of an instance, as the library computes them (names and encoding as in the split files). */
export function solve(task: Task, inst: Instance): Outputs {
	const x = inst.inputs;
	if (isGraph(task)) {
		const n = inst.n;
		const A = f64(x.A, 'A');
		if (A.length !== n * n) throw new Error(`A has ${A.length} entries for n = ${n}`);
		const s = () => scalar(x.s, 's');
		switch (task) {
			case 'bfs':
				return bfs(A, n, s());
			case 'bellman_ford':
				return bellmanFord(A, n, s());
			case 'dijkstra':
				return dijkstra(A, n, s());
			case 'dag_shortest_paths':
				return dagShortestPaths(A, n, s());
			case 'floyd_warshall':
				return floydWarshall(A, n);
			case 'mst_prim':
				return mstPrim(A, n, s());
			case 'topological_sort':
				return topologicalSort(A, n);
			case 'dfs':
				return dfs(A, n);
			case 'strongly_connected_components':
				return stronglyConnectedComponents(A, n);
			case 'articulation_points':
				return articulationPoints(A, n);
			case 'bridges':
				return bridges(A, n);
			case 'mst_kruskal':
				return mstKruskal(A, n);
		}
	}
	switch (task) {
		case 'insertion_sort':
			return insertionSort(f64(x.key, 'key'));
		case 'bubble_sort':
			return bubbleSort(f64(x.key, 'key'));
		case 'heapsort':
			return heapsort(f64(x.key, 'key'));
		case 'quicksort':
			return quicksort(f64(x.key, 'key'));
		case 'minimum':
			return minimum(f64(x.key, 'key'));
		case 'quickselect':
			return quickselect(f64(x.key, 'key'));
		case 'binary_search':
			return binarySearch(f64(x.key, 'key'), scalar(x.target, 'target'));
		case 'find_maximum_subarray_kadane':
			return kadane(f64(x.key, 'key'));
		case 'matrix_chain_order':
			return matrixChainOrder(f64(x.p, 'p'));
		case 'optimal_bst':
			return optimalBst(f64(x.p, 'p'), f64(x.q, 'q'));
		case 'lcs_length':
			return lcsLength(...twoStrings(inst));
		case 'activity_selector':
			return activitySelector(f64(x.s, 's'), f64(x.f, 'f'));
		case 'task_scheduling':
			return taskScheduling(f64(x.d, 'd'), f64(x.w, 'w'));
		case 'naive_string_matcher':
			return naiveStringMatcher(...twoStrings(inst));
		case 'kmp_matcher':
			return kmpMatcher(...twoStrings(inst));
		case 'segments_intersect':
			return segmentsIntersect(f64(x.x, 'x'), f64(x.y, 'y'));
		case 'graham_scan':
			return grahamScan(f64(x.x, 'x'), f64(x.y, 'y'));
		case 'jarvis_march':
			return jarvisMarch(f64(x.x, 'x'), f64(x.y, 'y'));
	}
	throw new Error(`unknown task ${task as string}`);
}

/** Positions of the outputs that differ from `expected` (exact comparison; [] when equal). */
export function diffOutputs(got: Outputs, expected: Outputs): string[] {
	const bad: string[] = [];
	for (const [name, e] of Object.entries(expected)) {
		const g = got[name];
		if (typeof e === 'number' || typeof g === 'number' || g === undefined) {
			if (g !== e) bad.push(`${name}: ${String(g)} != ${String(e)}`);
			continue;
		}
		if (g.length !== e.length) {
			bad.push(`${name}: length ${g.length} != ${e.length}`);
			continue;
		}
		for (let i = 0; i < e.length; i++) if (g[i] !== e[i]) bad.push(`${name}[${i}]: ${g[i]} != ${e[i]}`);
	}
	return bad;
}
