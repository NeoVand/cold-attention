/**
 * Positions for a graph's nodes by stress majorization (SMACOF) on its shortest-path distances (edge weights where the
 * task has them, hops otherwise; directions ignored): nodes sit as far apart as the graph says they are, so trees
 * from a source radiate and clusters stay together. Deterministic: starts on a circle, fixed iterations.
 */
export function stressLayout(n: number, A: ArrayLike<number>, adj: ArrayLike<number>, weighted: boolean, iters = 160): [number, number][] {
	const INF = 1e9;
	const d = new Float64Array(n * n).fill(INF);
	for (let u = 0; u < n; u++) {
		d[u * n + u] = 0;
		for (let v = 0; v < n; v++) {
			if (u === v) continue;
			const e = adj[u * n + v] > 0 || adj[v * n + u] > 0;
			if (!e) continue;
			const w = weighted ? Math.max(Math.min(A[u * n + v] || INF, A[v * n + u] || INF), 1e-3) : 1;
			d[u * n + v] = Math.min(d[u * n + v], w);
			d[v * n + u] = Math.min(d[v * n + u], w);
		}
	}
	for (let k = 0; k < n; k++)
		for (let u = 0; u < n; u++) {
			const duk = d[u * n + k];
			if (duk >= INF) continue;
			for (let v = 0; v < n; v++) {
				const s = duk + d[k * n + v];
				if (s < d[u * n + v]) d[u * n + v] = s;
			}
		}
	let max = 0;
	for (let i = 0; i < n * n; i++) if (d[i] < INF) max = Math.max(max, d[i]);
	if (max === 0) max = 1;
	for (let i = 0; i < n * n; i++) if (d[i] >= INF) d[i] = max * 1.3;
	const x = new Float64Array(n);
	const y = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		const a = (2 * Math.PI * i) / n;
		x[i] = Math.cos(a) * max * 0.5;
		y[i] = Math.sin(a) * max * 0.5;
	}
	for (let it = 0; it < iters; it++) {
		const nx = new Float64Array(n);
		const ny = new Float64Array(n);
		for (let i = 0; i < n; i++) {
			let sx = 0;
			let sy = 0;
			let ws = 0;
			for (let j = 0; j < n; j++) {
				if (i === j) continue;
				const dij = d[i * n + j];
				const w = 1 / (dij * dij);
				const dx = x[i] - x[j];
				const dy = y[i] - y[j];
				const r = Math.hypot(dx, dy) || 1e-9;
				sx += w * (x[j] + (dij * dx) / r);
				sy += w * (y[j] + (dij * dy) / r);
				ws += w;
			}
			nx[i] = sx / ws;
			ny[i] = sy / ws;
		}
		x.set(nx);
		y.set(ny);
	}
	// normalise into [0, 1]^2, keeping the aspect ratio
	let x0 = Infinity,
		x1 = -Infinity,
		y0 = Infinity,
		y1 = -Infinity;
	for (let i = 0; i < n; i++) {
		x0 = Math.min(x0, x[i]);
		x1 = Math.max(x1, x[i]);
		y0 = Math.min(y0, y[i]);
		y1 = Math.max(y1, y[i]);
	}
	const s = Math.max(x1 - x0, y1 - y0) || 1;
	const ox = (1 - (x1 - x0) / s) / 2;
	const oy = (1 - (y1 - y0) / s) / 2;
	return Array.from({ length: n }, (_, i) => [ox + (x[i] - x0) / s, oy + (y[i] - y0) / s]);
}

/** Children lists of a forest given parent pointers (roots point to themselves). */
function children(n: number, parent: ArrayLike<number>): { kids: number[][]; roots: number[] } {
	const kids: number[][] = Array.from({ length: n }, () => []);
	const roots: number[] = [];
	for (let i = 0; i < n; i++) {
		const p = parent[i];
		if (p === i || p < 0 || p >= n) roots.push(i);
		else kids[p].push(i);
	}
	return { kids, roots };
}

/**
 * A radial tidy tree: the root at the centre, each node on the ring of its depth, each subtree in a wedge as wide as its
 * share of the leaves (children in index order). A forest: the root `first` at the centre when it is the only tree,
 * otherwise the roots on the first ring. Nodes alone in their tree (not reached from the source) go on an outer ring,
 * outside the tree. Coordinates in [0, 1]^2 for a box of aspect w / h.
 */
export function radialTree(n: number, parent: ArrayLike<number>, first: number, aspect: number): [number, number][] {
	const { kids, roots } = children(n, parent);
	const singles = roots.filter((r) => r !== first && kids[r].length === 0);
	const order = [first, ...roots.filter((r) => r !== first && kids[r].length > 0)].filter((r) => roots.includes(r));
	const virtualRoot = order.length !== 1;
	const inTree = new Uint8Array(n);
	const leaves = new Float64Array(n);
	const depth = new Int32Array(n);
	const count = (v: number, d: number): number => {
		inTree[v] = 1;
		depth[v] = d;
		leaves[v] = kids[v].length ? kids[v].reduce((a, c) => a + count(c, d + 1), 0) : 1;
		return leaves[v];
	};
	for (const r of order) count(r, virtualRoot ? 1 : 0);
	const m = n - singles.length;
	let maxDepth = 1;
	for (let v = 0; v < n; v++) if (inTree[v]) maxDepth = Math.max(maxDepth, depth[v]);
	const angle = new Float64Array(n);
	const place = (v: number, a0: number, a1: number) => {
		angle[v] = (a0 + a1) / 2;
		let a = a0;
		for (const c of kids[v]) {
			const w = ((a1 - a0) * leaves[c]) / leaves[v];
			place(c, a, a + w);
			a += w;
		}
	};
	const total = order.reduce((a, r) => a + leaves[r], 0);
	let a = -Math.PI / 2;
	for (const r of order) {
		const w = (2 * Math.PI * leaves[r]) / total;
		place(r, a, a + w);
		a += w;
	}
	// radii: equal-area rings, the ring of depth d at the radius whose disc holds the nodes up to that depth, so the
	// nodes spread evenly however deep one branch runs; the tree keeps clear of the outer ring when there is one
	const R = singles.length ? 0.39 : 0.48;
	const per = new Float64Array(maxDepth + 2);
	for (let v = 0; v < n; v++) if (inTree[v]) per[depth[v]]++;
	const cum = new Float64Array(maxDepth + 2);
	for (let d = 0, c = 0; d <= maxDepth; d++) {
		cum[d] = c + per[d] / 2;
		c += per[d];
	}
	const rad = (d: number) => (d === 0 && !virtualRoot ? 0 : R * Math.sqrt(cum[d] / Math.max(1, m)));
	const radius = new Float64Array(n);
	for (let v = 0; v < n; v++) if (inTree[v]) radius[v] = rad(depth[v]);
	singles.forEach((v, i) => {
		angle[v] = -Math.PI / 2 + (2 * Math.PI * (i + 0.5)) / singles.length;
		radius[v] = 0.48;
	});
	return Array.from({ length: n }, (_, v) => [
		0.5 + (radius[v] * Math.cos(angle[v])) / Math.max(aspect, 1),
		0.5 + radius[v] * Math.sin(angle[v]) * Math.min(aspect, 1)
	]);
}

/** Nodes around a circle in the order given (DFS: discovery order, so tree edges are short and back-tracks are chords). */
function circle(order: number[], n: number, aspect: number): [number, number][] {
	const pos: [number, number][] = Array(n);
	order.forEach((v, i) => {
		const a = -Math.PI / 2 + (2 * Math.PI * i) / n;
		pos[v] = [0.5 + (0.47 * Math.cos(a)) / Math.max(aspect, 1), 0.5 + 0.47 * Math.sin(a) * Math.min(aspect, 1)];
	});
	return pos;
}

/** Preorder of a forest (roots in index order, children in index order). */
function preorder(n: number, parent: ArrayLike<number>): number[] {
	const { kids, roots } = children(n, parent);
	const out: number[] = [];
	const walk = (v: number) => {
		out.push(v);
		for (const c of kids[v]) walk(c);
	};
	for (const r of roots) walk(r);
	for (let i = 0; i < n; i++) if (!out.includes(i)) out.push(i);
	return out;
}

/** Parent pointers of a spanning forest given as a symmetric 0/1 edge mask, rooted at each component's centre. */
function forestFromMask(n: number, mask: ArrayLike<number>): { parent: Int32Array; centre: number } {
	const nb: number[][] = Array.from({ length: n }, () => []);
	for (let u = 0; u < n; u++) for (let v = 0; v < n; v++) if (u !== v && mask[u * n + v] === 1) nb[u].push(v);
	const bfs = (s: number) => {
		const d = new Int32Array(n).fill(-1);
		const par = new Int32Array(n).fill(-1);
		d[s] = 0;
		par[s] = s;
		const q = [s];
		for (let i = 0; i < q.length; i++) for (const w of nb[q[i]]) if (d[w] < 0) { d[w] = d[q[i]] + 1; par[w] = q[i]; q.push(w); }
		return { d, par, q };
	};
	const parent = new Int32Array(n).fill(-1);
	let centre = 0;
	let biggest = 0;
	for (let s = 0; s < n; s++) {
		if (parent[s] >= 0) continue;
		// the component's centre: the middle of a longest path (two sweeps)
		const a = bfs(s);
		const far = a.q.reduce((x, y) => (a.d[y] > a.d[x] ? y : x), s);
		const b = bfs(far);
		const far2 = b.q.reduce((x, y) => (b.d[y] > b.d[x] ? y : x), far);
		let mid = far2;
		for (let k = 0; k < Math.floor(b.d[far2] / 2); k++) mid = b.par[mid];
		const c = bfs(mid);
		for (const v of c.q) parent[v] = c.par[v];
		if (c.q.length > biggest) {
			biggest = c.q.length;
			centre = mid;
		}
	}
	return { parent, centre };
}

/**
 * The layout a task is drawn with, for a box of aspect w / h. Tree tasks (a node points to its parent): a radial tree
 * from the source by depth. DFS: around a circle in discovery order. Kruskal: a radial tree of the true spanning tree from
 * its centre. Topological sort: the nodes along their true order. Other graph tasks: stress on the graph itself.
 */
export function taskLayout(task: string, n: number, A: ArrayLike<number>, adj: ArrayLike<number>, truth: Record<string, ArrayLike<number>>, aspect = 1): [number, number][] {
	if (['bfs', 'bellman_ford', 'dijkstra', 'dag_shortest_paths', 'mst_prim'].includes(task)) {
		const s = truth.s ? truth.s[0] : 0;
		return radialTree(n, truth.pi, s, aspect);
	}
	if (task === 'dfs') return circle(preorder(n, truth.pi), n, aspect);
	if (task === 'mst_kruskal') {
		const { parent, centre } = forestFromMask(n, truth.in_mst);
		return radialTree(n, parent, centre, aspect);
	}
	if (task === 'topological_sort') {
		const next = truth.topo;
		const order: number[] = [];
		let at = truth.topo_head[0];
		const seen = new Set<number>();
		while (at >= 0 && !seen.has(at) && order.length < n) {
			order.push(at);
			seen.add(at);
			at = next[at];
		}
		for (let i = 0; i < n; i++) if (!seen.has(i)) order.push(i);
		const pos: [number, number][] = Array(n);
		order.forEach((v, r) => {
			const t = n === 1 ? 0.5 : r / (n - 1);
			pos[v] = [t, 0.5 + 0.18 * Math.sin(t * Math.PI * 2)];
		});
		return pos;
	}
	return stressLayout(n, A, adj, task === 'floyd_warshall');
}
