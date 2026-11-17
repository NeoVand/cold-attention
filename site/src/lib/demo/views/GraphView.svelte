<script lang="ts">
	/**
	 * A graph task, in the paper's figure language: ink is the input (the graph, light), the block's answer is cobalt for
	 * a cold block (exact: numbers compared, pair numbers) and rose for the general block, and the
	 * true answer is dashed ink wherever it differs. Click a node, then another, to pick a pair (u, v): the two are filled
	 * in ink, the witnesses k of the chosen channel pair are ringed (the block's colour: the first channel, with their
	 * routes u -> k -> v; grey, dashed: the second), and the rest fades back. Click the background to clear.
	 */
	import type { Arrays } from '../../model/inputs';
	import type { PairInfo } from '../worker';
	import { taskLayout } from '../layout';
	import { witnessSets } from '../witness';
	import { C, blockColour } from '../theme';
	import { arrow, headFor, link } from '../arrows';

	let {
		task,
		block,
		n,
		arrays,
		pred,
		sel,
		pending,
		pair,
		chPair,
		onitem,
		onclear
	}: {
		task: string;
		block: string;
		n: number;
		arrays: Arrays;
		pred: Record<string, Int32Array> | null;
		sel: { u: number; v: number } | null;
		pending: number | null;
		pair: PairInfo | null;
		chPair: number;
		onitem: (k: number) => void;
		onclear: () => void;
	} = $props();

	const S = 1000;
	const SH = 700;
	const PAD = 40;
	const ans = $derived(blockColour(block).c);
	const A = $derived(arrays.A.data);
	const adj = $derived(arrays.adj.data);
	const truthArrays = $derived(Object.fromEntries(Object.entries(arrays).map(([k, v]) => [k, v.data])));
	const BW = S - 2 * PAD - 60;
	const BH = SH - 2 * PAD;
	const pos = $derived(taskLayout(task, n, A, adj, truthArrays, BW / BH).map(([x, y]) => [PAD + 30 + x * BW, PAD + y * BH] as [number, number]));
	/** laid out as a tree (radial): input edges as faint hairlines, the answer as straight arrows */
	const tree = $derived(['bfs', 'bellman_ford', 'dijkstra', 'dag_shortest_paths', 'mst_prim', 'mst_kruskal'].includes(task));
	const ptrBend = $derived(tree ? 0 : task === 'dfs' ? 0.22 : 0.16);
	const r = $derived(Math.max(5, Math.min(14, 170 / n)));
	const labels = $derived(n <= 40);
	const source = $derived(arrays.s ? arrays.s.data[0] : -1);

	/** edges once per unordered pair; directed when only one way */
	const edges = $derived.by(() => {
		const out: { u: number; v: number; dir: boolean }[] = [];
		for (let u = 0; u < n; u++)
			for (let v = u + 1; v < n; v++) {
				const a = adj[u * n + v] > 0;
				const b = adj[v * n + u] > 0;
				if (a && b) out.push({ u, v, dir: false });
				else if (a) out.push({ u, v, dir: true });
				else if (b) out.push({ u: v, v: u, dir: true });
			}
		return out;
	});
	const ptrName = $derived(task === 'topological_sort' ? 'topo' : task === 'strongly_connected_components' ? 'scc_id' : 'pi');
	const hasPtr = $derived(['bfs', 'dfs', 'bellman_ford', 'dijkstra', 'dag_shortest_paths', 'mst_prim', 'topological_sort', 'strongly_connected_components'].includes(task));
	const truePtr = $derived(hasPtr ? arrays[ptrName].data : null);
	const predPtr = $derived(hasPtr && pred ? pred[ptrName] : null);
	const edgeMask = $derived(task === 'bridges' ? 'is_bridge' : task === 'mst_kruskal' ? 'in_mst' : null);
	const trueEdge = $derived(edgeMask ? arrays[edgeMask].data : null);
	const predEdge = $derived(edgeMask && pred ? pred[edgeMask] : null);
	const trueCut = $derived(task === 'articulation_points' ? arrays.is_cut.data : null);
	const predCut = $derived(task === 'articulation_points' && pred ? pred.is_cut : null);
	const head = $derived(task === 'topological_sort' && pred ? pred.topo_head[0] : -1);
	const trueHead = $derived(task === 'topological_sort' ? arrays.topo_head.data[0] : -1);
	const wit = $derived(sel ? witnessSets(pair, chPair, n) : null);
	/** with a pair picked, everything but the pair and its witnesses fades back */
	const focus = $derived(sel !== null);
	const involved = (i: number) => !sel || i === sel.u || i === sel.v || !!wit?.a.includes(i) || !!wit?.b.includes(i);
	/** the first channel's routes u -> k -> v, through the witnesses other than u and v (at most 12) */
	const routes = $derived(sel && wit ? wit.a.filter((k) => k !== sel.u && k !== sel.v).slice(0, 12) : []);

	/** a gently bent path from a to b, ending short of b's circle */
	function arc(a: number, b: number, bend = 0.18, shorten = 0): string {
		const [x1, y1] = pos[a];
		const [x2, y2] = pos[b];
		const dx = x2 - x1;
		const dy = y2 - y1;
		const L = Math.hypot(dx, dy) || 1;
		const ex = x2 - (dx / L) * shorten;
		const ey = y2 - (dy / L) * shorten;
		const mx = (x1 + ex) / 2 - dy * bend;
		const my = (y1 + ey) / 2 + dx * bend;
		return `M${x1.toFixed(1)},${y1.toFixed(1)} Q${mx.toFixed(1)},${my.toFixed(1)} ${ex.toFixed(1)},${ey.toFixed(1)}`;
	}

	/** each node's drawn radius (the selected pair's two nodes are larger) */
	const rad = (i: number) => (sel !== null && (sel.u === i || sel.v === i) ? r * 1.25 : r);
	const hl = $derived(headFor(r));
	/** an arrow from node a to node b, clipped at both circles */
	const arr = (a: number, b: number, bend: number, head = hl) => arrow(pos[a], pos[b], rad(a), rad(b), bend, head);

</script>

<svg viewBox="0 0 {S} {SH}" class="h-full w-full select-none" role="img" aria-label="graph">
	<rect width={S} height={SH} fill="transparent" role="button" tabindex="-1" aria-label="clear the pair" onclick={onclear} onkeydown={(e) => e.key === 'Escape' && onclear()} />
	<!-- the input graph: hairlines (faint where the layout is the answer's tree) -->
	<g stroke={C.mist} stroke-width={tree || task === 'dfs' ? 0.45 : n > 40 ? 0.5 : 0.9} opacity={task === 'dfs' ? (n > 40 ? 0.14 : 0.35) : tree ? (n > 40 ? 0.28 : 0.45) : n > 40 ? 0.45 : 0.75} fill="none">
		{#each edges as e (e.u * n + e.v)}
			{@const a = e.dir && n <= 40 && !tree && task !== 'dfs' ? arr(e.u, e.v, 0.06, hl * 0.8) : null}
			{#if a}
				<path d={a.shaft} />
				<path d={a.head} fill={C.mist} stroke="none" />
			{:else}
				<path d={arc(e.u, e.v, e.dir && !tree && task !== 'dfs' ? 0.06 : 0)} />
			{/if}
		{/each}
	</g>


	<!-- the answer: faded back while a pair is picked -->
	<g class="fade" opacity={focus ? 0.25 : 1} style:filter={focus ? 'grayscale(1)' : undefined}>
	<!-- edge answers: the block's (solid, its colour; dashed where it adds an edge), the true one (dashed ink) where it differs -->
	{#if trueEdge}
		<g fill="none" stroke-linecap="round">
			{#each edges as e (e.u * n + e.v)}
				{@const t = trueEdge[e.u * n + e.v] === 1}
				{@const p = predEdge ? predEdge[e.u * n + e.v] === 1 : false}
				{#if t && !p && predEdge}<path d={arc(e.u, e.v, 0)} stroke={C.ink} stroke-width="1.6" stroke-dasharray="5 4" />{/if}
				{#if p}<path d={arc(e.u, e.v, 0)} stroke={ans} stroke-width="2.6" stroke-dasharray={t ? undefined : '6 5'} />{/if}
			{/each}
		</g>
	{/if}

	<!-- pointer answers: each node to its parent (or the next node, or its component's representative) -->
	{#if truePtr}
		<g fill="none" stroke-linecap="round">
			{#each Array.from({ length: n }, (_, i) => i) as i (i)}
				{@const t = truePtr[i]}
				{@const p = predPtr ? predPtr[i] : -1}
				{#if t !== i && t >= 0 && p !== t && predPtr}
					{@const a = arr(i, t, ptrBend === 0 ? 0.12 : ptrBend)}
					{#if a}
						<path d={a.shaft} stroke={C.ink} stroke-width="1.5" stroke-dasharray="5 4" />
						<path d={a.head} fill={C.ink} stroke="none" />
					{/if}
				{/if}
				{#if p >= 0 && p !== i && p < n}
					{@const a = arr(i, p, ptrBend)}
					{#if a}
						<path d={a.shaft} stroke={ans} stroke-width={tree ? 2 : 2.2} />
						<path d={a.head} fill={ans} stroke="none" />
					{/if}
				{/if}
			{/each}
		</g>
	{/if}

	</g>

	<!-- the first channel's witnesses: routes u -> k -> v, clipped at the nodes -->
	{#if sel}
		<g fill="none" stroke={ans} stroke-width="1.8" stroke-linecap="round" opacity="0.75">
			{#each routes as k (k)}
				{@const a = link(pos[sel.u], pos[k], rad(sel.u), rad(k), 0.06)}
				{@const b = link(pos[k], pos[sel.v], rad(k), rad(sel.v), 0.06)}
				{#if a}<path d={a} />{/if}
				{#if b}<path d={b} />{/if}
			{/each}
		</g>
	{/if}

	<!-- nodes -->
	{#each pos as [x, y], i (i)}
		{@const cut = predCut ? predCut[i] === 1 : false}
		{@const tcut = trueCut ? trueCut[i] === 1 : false}
		{@const wrong = (truePtr && predPtr && truePtr[i] !== predPtr[i]) || (trueCut && predCut && cut !== tcut)}
		{@const isSel = sel !== null && (sel.u === i || sel.v === i)}
		{@const wa = wit?.a.includes(i)}
		{@const wb = wit?.b.includes(i)}
		<g transform="translate({x},{y})" class="fade cursor-pointer" role="button" tabindex="-1" aria-label="node {i}" opacity={focus && !involved(i) ? 0.4 : 1}
			style:filter={focus && !involved(i) ? 'grayscale(1)' : undefined} onclick={() => onitem(i)} onkeydown={(e) => e.key === 'Enter' && onitem(i)}>
			{#if tcut && !cut && predCut}<circle r={r + 3.5} fill="none" stroke={C.ink} stroke-width="1.6" stroke-dasharray="3 3" />{/if}
			<circle
				r={isSel ? r * 1.25 : r}
				fill={isSel ? C.ink : cut || i === head ? ans : C.paper}
				stroke={isSel ? C.ink : wa ? ans : wb ? C.blueGrey : wrong && !focus ? C.ink : i === source || i === trueHead ? C.ink : C.blueGrey}
				stroke-width={isSel ? 0 : wa ? 2.8 : wb ? 2.2 : wrong && !focus ? 2.2 : i === source || i === trueHead ? 2.2 : 1.1}
				stroke-dasharray={wb && !wa ? '3 2' : wrong && !focus && !wa && !wb ? '3 2' : undefined}
			/>
			{#if pending === i}<circle r={r + 3.5} fill="none" stroke={C.ink} stroke-width="1.8" />{/if}
			{#if labels || isSel}
				<text dy="0.35em" text-anchor="middle" font-size={r * (isSel ? 1.05 : 0.95)} fill={isSel || cut || i === head ? C.paper : C.ink} class="num" pointer-events="none">{i}</text>
			{/if}
		</g>
	{/each}
	{#if sel}
		<text x={pos[sel.u][0]} y={pos[sel.u][1] - r * 1.25 - 8} text-anchor="middle" font-size="22" font-weight="600" fill={C.ink}>u</text>
		<text x={pos[sel.v][0]} y={pos[sel.v][1] - r * 1.25 - 8} text-anchor="middle" font-size="22" font-weight="600" fill={C.ink}>v</text>
	{/if}
</svg>

<style>
	.fade {
		transition: opacity 220ms var(--ease);
	}
</style>
