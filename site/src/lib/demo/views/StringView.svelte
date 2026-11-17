<script lang="ts">
	/**
	 * String matching: the haystack as a row of letters, the needle under it at the position the block names (in the
	 * block's colour) and at the true first match (ink) when they differ. Click two letters to pick a pair; its witnesses
	 * are dotted.
	 */
	import type { Arrays } from '../../model/inputs';
	import type { PairInfo } from '../worker';
	import { witnessSets } from '../witness';
	import { C, blockColour } from '../theme';

	let {
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

	const ansLight = $derived(blockColour(block).light);

	const items = $derived(Array.from({ length: n }, (_, i) => i));
	const hay = $derived(items.filter((i) => arrays.x_string.data[i] === 0));
	const needle = $derived(items.filter((i) => arrays.x_string.data[i] === 1));
	const letter = (i: number) => {
		const k = arrays.x_key.data;
		for (let c = 0; c < 4; c++) if (k[i * 4 + c]) return 'abcd'[c];
		return '?';
	};
	const W = 1000;
	const cw = $derived(Math.min(40, (W - 40) / hay.length));
	const truth = $derived(arrays.match.data[0]);
	const guess = $derived(pred ? pred.match[0] : -1);
	const wit = $derived(sel ? witnessSets(pair, chPair, n) : null);
	const focus = $derived(sel !== null);
	const involved = (i: number) => !sel || i === sel.u || i === sel.v || !!wit?.a.includes(i) || !!wit?.b.includes(i);
	const xh = (r: number) => 20 + r * cw;
</script>

{#snippet glyph(i: number, x: number, y: number, fill: string)}
	{@const isSel = sel !== null && (sel.u === i || sel.v === i)}
	<g role="button" tabindex="-1" aria-label="letter {i}" class="fade cursor-pointer" opacity={focus && !involved(i) ? 0.4 : 1} style:filter={focus && !involved(i) ? 'grayscale(1)' : undefined}
		onclick={() => onitem(i)} onkeydown={(e) => e.key === 'Enter' && onitem(i)}>
		<rect x={x + 1} {y} width={cw - 2} height={cw * 1.15} rx="2" {fill} stroke={isSel || pending === i ? C.ink : 'none'} stroke-width="2" />
		{#if isSel}<text x={x + cw / 2} y={y + cw * 1.15 + 16} text-anchor="middle" font-size="15" font-weight="600" fill={C.ink}>{sel!.u === i ? 'u' : 'v'}</text>{/if}
		<text x={x + cw / 2} y={y + cw * 0.6} dy="0.35em" text-anchor="middle" font-size={cw * 0.55} font-family="var(--font-mono)" fill={C.ink} pointer-events="none">{letter(i)}</text>
		{#if wit?.a.includes(i)}<circle cx={x + cw / 2} cy={y - 7} r="3.5" fill={blockColour(block).c} />{/if}
		{#if wit?.b.includes(i)}<circle cx={x + cw / 2} cy={y - 7} r="6.5" fill="none" stroke={C.blueGrey} stroke-width="1.8" stroke-dasharray="3 2" />{/if}
	</g>
{/snippet}

<svg viewBox="0 0 {W} 330" class="h-full w-full select-none" role="img" aria-label="strings">
	<rect width={W} height="330" fill="transparent" role="button" tabindex="-1" aria-label="clear the pair" onclick={onclear} onkeydown={(e) => e.key === 'Escape' && onclear()} />
	<text x="20" y="40" font-size="14" fill={C.inkSoft}>haystack</text>
	{#each hay as i, r (i)}
		{@render glyph(i, xh(r), 60, guess >= 0 && r >= guess && r < guess + needle.length ? ansLight : C.tint)}
	{/each}
	<text x="20" y="150" font-size="14" fill={C.inkSoft}>needle, where the block puts it</text>
	{#if guess >= 0}
		{#each needle as i, r (i)}{@render glyph(i, xh(Math.min(guess, hay.length) + r), 170, ansLight)}{/each}
	{/if}
	{#if guess !== truth}
		<text x="20" y="255" font-size="14" fill={C.inkSoft}>needle, at the true first match</text>
		{#each needle as i, r (i)}
			<rect x={xh(Math.min(truth, hay.length) + r) + 1} y="270" width={cw - 2} height={cw * 1.15} rx="2" fill="none" stroke={C.ink} stroke-width="1.6" stroke-dasharray="4 3" />
			<text x={xh(Math.min(truth, hay.length) + r) + cw / 2} y={270 + cw * 0.6} dy="0.35em" text-anchor="middle" font-size={cw * 0.55} font-family="var(--font-mono)" fill={C.ink}>{letter(i)}</text>
		{/each}
	{/if}
</svg>

<style>
	.fade {
		transition: opacity 220ms var(--ease);
	}
</style>
