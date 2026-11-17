<script lang="ts">
	/**
	 * Greedy tasks. Activity selector: each activity is a bar from its start to its finish. Task scheduling: each unit
	 * task is a bar at its deadline, as tall as its weight. Filled in the block's colour: chosen by the block; dashed ink
	 * outline: where the true answer differs. Click two bars to pick a pair; its witnesses are marked.
	 */
	import type { Arrays } from '../../model/inputs';
	import type { PairInfo } from '../worker';
	import { witnessSets } from '../witness';
	import { C, blockColour } from '../theme';

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

	/** the block's answer: cobalt for a cold block (exact), rose for the general block (one per task or shared) */
	const ans = $derived(blockColour(block).c);
	const ansLight = $derived(blockColour(block).light);

	const W = 1000;
	const act = $derived(task === 'activity_selector');
	const rowH = $derived(Math.min(26, 520 / n));
	const H = $derived(act ? 40 + n * rowH + 20 : 560);
	const truth = $derived(arrays.selected.data);
	const wit = $derived(sel ? witnessSets(pair, chPair, n) : null);
	const focus = $derived(sel !== null);
	const involved = (i: number) => !sel || i === sel.u || i === sel.v || !!wit?.a.includes(i) || !!wit?.b.includes(i);
	/** task scheduling: tasks stacked in columns by deadline */
	const stacks = $derived.by(() => {
		if (act) return [];
		const d = arrays.x_d.data as Float64Array;
		const maxD = Math.max(...d);
		const height = new Float64Array(maxD + 2);
		return Array.from({ length: n }, (_, i) => {
			const col = d[i];
			const h = (arrays.x_w.data as Float64Array)[i];
			const y0 = height[col];
			height[col] += h;
			return { i, col, y0, h, maxD };
		});
	});
	const maxStack = $derived(stacks.length ? Math.max(1, ...stacks.map((s) => s.y0 + s.h)) : 1);
</script>

<svg viewBox="0 0 {W} {H}" class="h-full w-full select-none" role="img" aria-label="intervals">
	<rect width={W} height={H} fill="transparent" role="button" tabindex="-1" aria-label="clear the pair" onclick={onclear} onkeydown={(e) => e.key === 'Escape' && onclear()} />
	{#if act}
		{@const s = arrays.x_s.data as Float64Array}
		{@const f = arrays.x_f.data as Float64Array}
		{#each Array.from({ length: n }, (_, i) => i) as i (i)}
			{@const p = pred ? pred.selected[i] === 1 : false}
			{@const t = truth[i] === 1}
			{@const bx = 30 + s[i] * (W - 60)}
			{@const by = 30 + i * rowH + rowH * 0.18}
			{@const bwid = Math.max(2, (f[i] - s[i]) * (W - 60))}
			<g role="button" tabindex="-1" aria-label="activity {i}" class="fade cursor-pointer" opacity={focus && !involved(i) ? 0.4 : 1} style:filter={focus && !involved(i) ? 'grayscale(1)' : undefined}
				onclick={() => onitem(i)} onkeydown={(e) => e.key === 'Enter' && onitem(i)}>
				<rect x="20" y={30 + i * rowH} width={W - 40} height={rowH} fill="transparent" />
				<rect x={bx} y={by} width={bwid} height={rowH * 0.64} rx="2" fill={p ? (t ? ans : ansLight) : C.edge}
					stroke={t && !p && pred ? C.ink : 'none'} stroke-width="1.6" stroke-dasharray="4 3" />
				{#if (sel && (sel.u === i || sel.v === i)) || pending === i}
					<rect x={bx - 2.5} y={by - 2.5} width={bwid + 5} height={rowH * 0.64 + 5} rx="3" fill="none" stroke={C.ink} stroke-width="2" />
				{/if}
				{#if wit?.a.includes(i)}<circle cx="16" cy={30 + (i + 0.5) * rowH} r={Math.min(6, rowH * 0.3)} fill={ans} />{/if}
				{#if wit?.b.includes(i)}<circle cx="16" cy={30 + (i + 0.5) * rowH} r={Math.min(6, rowH * 0.3) + 3} fill="none" stroke={C.blueGrey} stroke-width="1.8" stroke-dasharray="3 2" />{/if}
				{#if sel && (sel.u === i || sel.v === i)}
					{@const right = bx + bwid + 26 < W}
					<text x={right ? bx + bwid + 10 : bx - 10} y={30 + (i + 0.5) * rowH} dy="0.35em" text-anchor={right ? 'start' : 'end'} font-size="15" font-weight="600" fill={C.ink}>{sel.u === i ? 'u' : 'v'}</text>
				{/if}
			</g>
		{/each}
		<text x="30" y="20" font-size="14" fill={C.inkSoft}>time →</text>
	{:else}
		{@const cw = (W - 60) / (stacks[0]?.maxD ?? n)}
		{#each stacks as b (b.i)}
			{@const p = pred ? pred.selected[b.i] === 1 : false}
			{@const t = truth[b.i] === 1}
			{@const bx = 30 + (b.col - 1) * cw + 2}
			{@const by = 500 - ((b.y0 + b.h) / maxStack) * 450}
			{@const bh = Math.max(1.5, (b.h / maxStack) * 450 - 1.5)}
			<g role="button" tabindex="-1" aria-label="task {b.i}" class="fade cursor-pointer" opacity={focus && !involved(b.i) ? 0.4 : 1} style:filter={focus && !involved(b.i) ? 'grayscale(1)' : undefined}
				onclick={() => onitem(b.i)} onkeydown={(e) => e.key === 'Enter' && onitem(b.i)}>
				<rect x={bx} y={by} width={cw - 4} height={bh} rx="1.5" fill={p ? (t ? ans : ansLight) : C.edge}
					stroke={t && !p && pred ? C.ink : 'none'} stroke-width="1.6" stroke-dasharray="4 3" />
				{#if (sel && (sel.u === b.i || sel.v === b.i)) || pending === b.i}
					<rect x={bx - 2} y={by - 2} width={cw} height={bh + 4} rx="3" fill="none" stroke={C.ink} stroke-width="2" />
				{/if}
				{#if wit?.a.includes(b.i)}<circle cx={30 + (b.col - 0.5) * cw} cy={500 - ((b.y0 + b.h / 2) / maxStack) * 450} r="4" fill={C.paper} stroke={ans} stroke-width="2.5" />{/if}
				{#if wit?.b.includes(b.i)}<circle cx={30 + (b.col - 0.5) * cw} cy={500 - ((b.y0 + b.h / 2) / maxStack) * 450} r="7" fill="none" stroke={C.blueGrey} stroke-width="2" stroke-dasharray="3 2" />{/if}
				{#if sel && (sel.u === b.i || sel.v === b.i)}<text x={30 + (b.col - 0.5) * cw} y={500 - ((b.y0 + b.h / 2) / maxStack) * 450} dy="0.35em" text-anchor="middle" font-size="14" font-weight="700" fill={C.ink}>{sel.u === b.i ? 'u' : 'v'}</text>{/if}
			</g>
		{/each}
		<line x1="26" x2={W - 26} y1="500" y2="500" stroke={C.mist} />
		<text x="30" y="528" font-size="14" fill={C.inkSoft}>deadline →   (bar height: weight)</text>
	{/if}
</svg>

<style>
	.fade {
		transition: opacity 220ms var(--ease);
	}
</style>
