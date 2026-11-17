<script lang="ts">
	/**
	 * An array task: one bar per number. Sorts: the bars stand where the block's pointers put them (each number just
	 * after the one it points to, the smallest first), and slide there as the loops arrive. Minimum, median, binary search
	 * and the maximum subarray: the bars in input order, the ones the block chooses filled. Everywhere the same rule: a
	 * bar the block gets right is filled in its colour, one it gets wrong is light, and a bar the true answer has where the
	 * block's differs is outlined in dashed ink. Click a bar, then another, to pick a pair: the two are outlined in ink, the
	 * witnesses are marked under the baseline, and the rest fades back. Click the background to clear.
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

	const ans = $derived(blockColour(block).c);
	const ansLight = $derived(blockColour(block).light);

	const kadane = $derived(task === 'find_maximum_subarray_kadane');
	const W = 1000;
	/** the maximum subarray has negative numbers too: its baseline sits higher, with room below */
	const H = $derived(kadane ? 500 : 420);
	const base = $derived(kadane ? 270 : 360);
	const amp = $derived(kadane ? 200 : 300);
	/** the witness marks' row, under everything */
	const wy = $derived(kadane ? H - 16 : base + 18);
	const key = $derived(Array.from(arrays.x_key.data as Float64Array));
	const lo = $derived(Math.min(0, ...key));
	const hi = $derived(Math.max(1, ...key));
	const bw = $derived((W - 40) / n);
	const x = (i: number) => 20 + (i + 0.5) * bw;
	const y = (v: number) => base - (v / (hi - lo)) * amp;
	const top = (v: number) => Math.min(y(v), base);
	const tall = (v: number) => Math.max(1.5, Math.abs(base - y(v)));
	const sorts = $derived(['insertion_sort', 'bubble_sort', 'heapsort', 'quicksort'].includes(task));

	/**
	 * Each item's slot in the order a pointer array gives: its hops down the chain to the item pointing to itself (the
	 * smallest), ties by index; items whose chain never ends there go last, by index.
	 */
	function slots(ptr: ArrayLike<number>): Int32Array {
		const depth = new Array<number>(n);
		for (let i = 0; i < n; i++) {
			let a = i;
			let d = 0;
			while (d <= n && a >= 0 && a < n && ptr[a] !== a) {
				a = ptr[a];
				d++;
			}
			depth[i] = d > n || a < 0 || a >= n ? n + 1 : d;
		}
		const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => depth[a] - depth[b] || a - b);
		const slot = new Int32Array(n);
		order.forEach((item, k) => (slot[item] = k));
		return slot;
	}
	const trueSlot = $derived(sorts ? slots(arrays.pred.data) : null);
	const slot = $derived(sorts && pred ? slots(pred.pred) : null);
	/** sorts: the true item in each slot where the block puts another */
	const missing = $derived.by(() => {
		if (!trueSlot || !slot) return [];
		const out: { k: number; item: number }[] = [];
		for (let i = 0; i < n; i++) if (trueSlot[i] !== slot[i]) out.push({ k: trueSlot[i], item: i });
		return out;
	});
	/** where bar i stands: its slot when sorting, else its input position */
	const sx = (i: number) => x(slot ? slot[i] : i);

	const one = $derived(task === 'minimum' ? 'min' : task === 'quickselect' ? 'median' : task === 'binary_search' ? 'return' : null);
	/** the bars the block chooses, and the true ones (single choices and the maximum subarray's run) */
	const chosen = (i: number) => {
		if (!pred) return false;
		if (one) return pred[one][0] === i;
		if (kadane) return pred.start[0] <= i && i <= pred.end[0];
		return false;
	};
	const truth = (i: number) => {
		if (one) return arrays[one].data[0] === i;
		if (kadane) return arrays.start.data[0] <= i && i <= arrays.end.data[0];
		return false;
	};
	function fill(i: number): string {
		if (sorts) return slot ? (trueSlot && trueSlot[i] !== slot[i] ? ansLight : ans) : C.edge;
		if (chosen(i)) return truth(i) ? ans : ansLight;
		return C.edge;
	}

	const wit = $derived(sel ? witnessSets(pair, chPair, n) : null);
	const focus = $derived(sel !== null);
	const involved = (i: number) => !sel || i === sel.u || i === sel.v || !!wit?.a.includes(i) || !!wit?.b.includes(i);
	const target = $derived(task === 'binary_search' ? (arrays.x_target.data[0] as number) : null);
	const mark = $derived(Math.min(6, bw * 0.28));
</script>

<svg viewBox="0 0 {W} {H}" class="h-full w-full select-none" role="img" aria-label="array">
	<rect width={W} height={H} fill="transparent" role="button" tabindex="-1" aria-label="clear the pair" onclick={onclear} onkeydown={(e) => e.key === 'Escape' && onclear()} />
	<line x1="16" x2={W - 16} y1={base} y2={base} stroke={C.mist} />

	{#if target !== null}
		<line x1="16" x2={W - 16} y1={y(target)} y2={y(target)} stroke={C.ink} stroke-dasharray="3 4" />
		<text x={W - 18} y={y(target) - 6} text-anchor="end" font-size="15" fill={C.inkSoft}>target</text>
	{/if}

	{#each key as v, i (i)}
		{@const isSel = sel !== null && (sel.u === i || sel.v === i)}
		<g class="bar fade cursor-pointer" role="button" tabindex="-1" aria-label="item {i}" style:transform="translateX({sx(i) - x(i)}px)"
			opacity={focus && !involved(i) ? 0.4 : 1} style:filter={focus && !involved(i) ? 'grayscale(1)' : undefined} onclick={() => onitem(i)} onkeydown={(e) => e.key === 'Enter' && onitem(i)}>
			<rect x={x(i) - bw / 2} y="10" width={bw} height={H - 20} fill="transparent" />
			<rect x={x(i) - bw * 0.36} y={top(v)} width={bw * 0.72} height={tall(v)} rx="1.5" fill={fill(i)} />
			{#if isSel || pending === i}
				<rect x={x(i) - bw * 0.36 - 2.5} y={top(v) - 2.5} width={bw * 0.72 + 5} height={tall(v) + 5} rx="3" fill="none" stroke={C.ink} stroke-width="2" />
			{/if}
			{#if isSel}
				<text x={x(i)} y={(v >= 0 ? top(v) : base) - 10} text-anchor="middle" font-size="18" font-weight="600" fill={C.ink}>{sel!.u === i ? 'u' : 'v'}</text>
			{/if}
			{#if wit?.a.includes(i)}<circle cx={x(i)} cy={wy} r={mark} fill={ans} />{/if}
			{#if wit?.b.includes(i)}<circle cx={x(i)} cy={wy} r={mark + 3} fill="none" stroke={C.blueGrey} stroke-width="1.8" stroke-dasharray="3 2" />{/if}
		</g>
	{/each}

	<!-- the true bar, dashed ink, where the block's answer differs -->
	<g class="fade" pointer-events="none" opacity={focus ? 0.2 : 1} fill="none" stroke={C.ink} stroke-width="1.4" stroke-dasharray="4 3">
		{#each missing as m (m.item)}
			<rect x={x(m.k) - bw * 0.36} y={top(key[m.item])} width={bw * 0.72} height={tall(key[m.item])} rx="1.5" />
		{/each}
		{#if !sorts && pred}
			{#each key as v, i (i)}
				{#if truth(i) && !chosen(i)}<rect x={x(i) - bw * 0.36} y={top(v)} width={bw * 0.72} height={tall(v)} rx="1.5" />{/if}
			{/each}
		{/if}
	</g>
</svg>

<style>
	.bar {
		transition:
			transform 420ms var(--ease),
			opacity 220ms var(--ease);
	}
	.fade {
		transition: opacity 220ms var(--ease);
	}
	@media (prefers-reduced-motion: reduce) {
		.bar {
			transition: none;
		}
	}
</style>
