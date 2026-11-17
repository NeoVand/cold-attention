<script lang="ts">
	/**
	 * Geometry. Convex hull: the points, the true hull's outline (ink), the points the block puts on the hull (in the
	 * block's colour; a dashed ring where it is wrong). Segments: the two segments and the block's verdict. Click two
	 * points to pick a pair; its witnesses are ringed.
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

	const S = 1000;
	const xs = $derived(Array.from(arrays.x_x.data as Float64Array));
	const ys = $derived(Array.from(arrays.x_y.data as Float64Array));
	const box = $derived.by(() => {
		const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
		const s = Math.max(x1 - x0, y1 - y0) || 1;
		return { x0, y0, s };
	});
	const px = (i: number) => 70 + ((xs[i] - box.x0) / box.s) * (S - 140);
	const py = (i: number) => S - 70 - ((ys[i] - box.y0) / box.s) * (S - 140);
	const hull = $derived(task !== 'segments_intersect');
	/** the true hull in order (by angle around the hull points' centroid) */
	const ring = $derived.by(() => {
		if (!hull) return [];
		const on = Array.from({ length: n }, (_, i) => i).filter((i) => arrays.in_hull.data[i] === 1);
		const cx = on.reduce((a, i) => a + xs[i], 0) / on.length;
		const cy = on.reduce((a, i) => a + ys[i], 0) / on.length;
		return on.sort((a, b) => Math.atan2(ys[a] - cy, xs[a] - cx) - Math.atan2(ys[b] - cy, xs[b] - cx));
	});
	const wit = $derived(sel ? witnessSets(pair, chPair, n) : null);
	const r = $derived(Math.max(5, Math.min(11, 160 / n)));
	const focus = $derived(sel !== null);
	const involved = (i: number) => !sel || i === sel.u || i === sel.v || !!wit?.a.includes(i) || !!wit?.b.includes(i);
	const verdict = (v: number) => (v === 1 ? 'they cross' : 'they do not cross');
</script>

<svg viewBox="0 0 {S} {S}" class="h-full w-full select-none" role="img" aria-label="points">
	<rect width={S} height={S} fill="transparent" role="button" tabindex="-1" aria-label="clear the pair" onclick={onclear} onkeydown={(e) => e.key === 'Escape' && onclear()} />
	{#if hull}
		<polygon points={ring.map((i) => `${px(i)},${py(i)}`).join(' ')} fill={C.tint} stroke={C.inkSoft} stroke-width="1.4" stroke-linejoin="round" />
	{:else}
		<line x1={px(0)} y1={py(0)} x2={px(1)} y2={py(1)} stroke={C.ink} stroke-width="3" stroke-linecap="round" />
		<line x1={px(2)} y1={py(2)} x2={px(3)} y2={py(3)} stroke={C.blueGrey} stroke-width="3" stroke-linecap="round" />
		<text x="40" y="44" font-size="22" fill={C.ink}>
			Block: <tspan fill={ans} font-weight="600">{pred ? verdict(pred.intersect[0]) : '…'}</tspan>
			· true: <tspan fill={C.ink} font-weight="600">{verdict(arrays.intersect.data[0])}</tspan>
		</text>
	{/if}
	{#each xs as _, i (i)}
		{@const p = hull && pred ? pred.in_hull[i] === 1 : false}
		{@const t = hull ? arrays.in_hull.data[i] === 1 : false}
		{@const isSel = sel !== null && (sel.u === i || sel.v === i)}
		<g role="button" tabindex="-1" aria-label="point {i}" class="fade cursor-pointer" opacity={focus && !involved(i) ? 0.4 : 1} style:filter={focus && !involved(i) ? 'grayscale(1)' : undefined}
			onclick={() => onitem(i)} onkeydown={(e) => e.key === 'Enter' && onitem(i)}>
			{#if wit?.a.includes(i)}<circle cx={px(i)} cy={py(i)} r={r + 6} fill="none" stroke={ans} stroke-width="2.4" />{/if}
			{#if wit?.b.includes(i)}<circle cx={px(i)} cy={py(i)} r={r + (wit?.a.includes(i) ? 10 : 6)} fill="none" stroke={C.blueGrey} stroke-width="2.4" stroke-dasharray="3 2" />{/if}
			<circle cx={px(i)} cy={py(i)} r={isSel ? r * 1.2 : r} fill={isSel ? C.ink : p ? (t ? ans : ansLight) : C.paper}
				stroke={isSel ? C.ink : t && !p && pred ? C.ink : C.blueGrey} stroke-width={t && !p && pred ? 2.2 : 1.2} stroke-dasharray={t && !p && pred && !isSel ? '3 2' : undefined} />
			{#if sel && (sel.u === i || sel.v === i)}<text x={px(i)} y={py(i) - r - 8} text-anchor="middle" font-size="20" font-weight="600" fill={C.ink}>{sel.u === i ? 'u' : 'v'}</text>{/if}
			{#if pending === i}<circle cx={px(i)} cy={py(i)} r={r + 3.5} fill="none" stroke={C.ink} stroke-width="1.8" />{/if}
		</g>
	{/each}
</svg>

<style>
	.fade {
		transition: opacity 220ms var(--ease);
	}
</style>
