<script lang="ts">
	/**
	 * A table task, one cell per pair (i, j): Floyd-Warshall's node before j on a shortest path from i; the best split of
	 * a matrix chain; the root of an optimal search tree; the arrows of the LCS table. A cell is light when the block's
	 * answer is right, tinted in the block's colour when it is wrong (the true answer shows on hover). Click a cell to pick
	 * that pair.
	 */
	import type { Arrays } from '../../model/inputs';
	import { C, blockColour } from '../theme';
	import { arrow } from '../arrows';

	let {
		task,
		block,
		n,
		arrays,
		pred,
		sel,
		onpick
	}: {
		task: string;
		block: string;
		n: number;
		arrays: Arrays;
		pred: Record<string, Int32Array> | null;
		sel: { u: number; v: number } | null;
		onpick: (u: number, v: number) => void;
	} = $props();

	const ans = $derived(blockColour(block).c);
	const ansLight = $derived(blockColour(block).light);

	const name = $derived(task === 'floyd_warshall' ? 'Pi' : task === 'matrix_chain_order' ? 's' : task === 'optimal_bst' ? 'root' : 'b');
	const truth = $derived(arrays[name].data);
	const lcs = $derived(task === 'lcs_length');
	/** LCS: rows are the first string, columns the second (cells (i, |x| + j)) */
	const xs = $derived(lcs ? Array.from({ length: n }, (_, i) => i).filter((i) => arrays.x_string.data[i] === 0) : []);
	const ys = $derived(lcs ? Array.from({ length: n }, (_, i) => i).filter((i) => arrays.x_string.data[i] === 1) : []);
	const rows = $derived(lcs ? xs : Array.from({ length: n }, (_, i) => i));
	const cols = $derived(lcs ? ys : Array.from({ length: n }, (_, i) => i));
	const used = (i: number, j: number) => (lcs ? true : task === 'floyd_warshall' ? true : j > i);
	const W = 1000;
	const cell = $derived(Math.min(56, (W - 70) / Math.max(rows.length, cols.length)));
	/** the table centred, with room on the left and top for the LCS strings' letters */
	const ox = $derived(Math.max(50, (W - cols.length * cell) / 2));
	const oy = 40;
	const letter = (i: number) => {
		const k = arrays.x_key.data;
		for (let c = 0; c < 4; c++) if (k[i * 4 + c]) return 'abcd'[c];
		return '?';
	};
	const ARROW = ['↖', '↑', '←', '·'];
	const show = (v: number) => (lcs ? (ARROW[v] ?? '') : String(v));
	/** an LCS entry drawn as an arrow in its cell: 0 up-left (a match), 1 up, 2 left */
	function lcsArrow(v: number, cx: number, cy: number) {
		const h = cell * 0.27;
		const d = v === 0 ? [-0.75, -0.75] : v === 1 ? [0, -1] : v === 2 ? [-1, 0] : null;
		if (!d) return null;
		return arrow([cx - d[0] * h, cy - d[1] * h], [cx + d[0] * h, cy + d[1] * h], 0, 0, 0, Math.max(4, cell * 0.2), 0);
	}
	let hover = $state<{ i: number; j: number } | null>(null);
</script>

<svg viewBox="0 0 {W} {oy + rows.length * cell + 60}" class="h-full w-full select-none" role="img" aria-label="table">
	{#if lcs}
		{#each rows as i, r (i)}<text x={ox - 10} y={oy + (r + 0.5) * cell} dy="0.35em" text-anchor="end" font-size={Math.min(18, cell * 0.6)} fill={C.inkSoft} font-family="var(--font-mono)">{letter(i)}</text>{/each}
		{#each cols as j, c (j)}<text x={ox + (c + 0.5) * cell} y={oy - 10} text-anchor="middle" font-size={Math.min(18, cell * 0.6)} fill={C.inkSoft} font-family="var(--font-mono)">{letter(j)}</text>{/each}
	{/if}
	{#each rows as i, r (i)}
		{#each cols as j, c (j)}
			{#if used(i, j)}
				{@const t = truth[i * n + j]}
				{@const p = pred ? pred[name][i * n + j] : -1}
				{@const ok = !pred || p === t}
				{@const isSel = sel && sel.u === i && sel.v === j}
				<g role="button" tabindex="-1" class="cursor-pointer"
					onclick={() => onpick(i, j)} onkeydown={(e) => e.key === 'Enter' && onpick(i, j)}
					onmouseenter={() => (hover = { i, j })} onmouseleave={() => (hover = null)}>
					<rect x={ox + c * cell + 0.5} y={oy + r * cell + 0.5} width={cell - 1} height={cell - 1} rx="1.5"
						fill={ok ? C.tint : ansLight} stroke={isSel ? C.ink : 'none'} stroke-width="2" />
					{#if cell >= 14 && pred && p >= 0 && lcs}
						{@const a = lcsArrow(p, ox + (c + 0.5) * cell, oy + (r + 0.5) * cell)}
						{#if a}
							<g pointer-events="none" stroke-linecap="round">
								<path d={a.shaft} stroke={ok ? C.inkSoft : ans} stroke-width={ok ? 1.3 : 1.8} fill="none" />
								<path d={a.head} fill={ok ? C.inkSoft : ans} />
							</g>
						{/if}
					{:else if cell >= 18 && pred && p >= 0}
						<text x={ox + (c + 0.5) * cell} y={oy + (r + 0.5) * cell} dy="0.35em" text-anchor="middle" font-size={Math.min(17, cell * 0.45)}
							fill={ok ? C.inkSoft : ans} font-weight={ok ? 400 : 600} class="num" pointer-events="none">{show(p)}</text>
					{/if}
				</g>
			{/if}
		{/each}
	{/each}
	{#if hover}
		{@const t = truth[hover.i * n + hover.j]}
		{@const p = pred ? pred[name][hover.i * n + hover.j] : -1}
		<text x={ox} y={oy + rows.length * cell + 32} font-size="17" fill={C.ink}>
			({hover.i}, {hover.j}): block <tspan fill={ans} font-weight="600">{show(p)}</tspan>, true <tspan fill={C.ink} font-weight="600">{show(t)}</tspan>
		</text>
	{/if}
</svg>
