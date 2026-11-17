<script lang="ts">
	/**
	 * The selected pair's count half at the loop shown: one cell per channel pair (32), showing the two witness counts
	 * and the bits handed on in the block's colour (full more, light same, pale fewer). Register channel pairs (compare-first, pair
	 * numbers) compare numbers at the witnesses rather than counts; they are marked with a dot. Click a cell to choose
	 * the channel pair whose witnesses are drawn.
	 */
	import type { PairInfo } from './worker';
	import { C, blockColour } from './theme';

	let { pair, chPair, registers, block, onchoose }: { pair: PairInfo | null; chPair: number; registers: boolean[]; block: string; onchoose: (c: number) => void } = $props();
	const bc = $derived(blockColour(block));
	const P = 32;
</script>

<div class="grid grid-cols-8 gap-[3px]">
	{#each Array.from({ length: P }, (_, i) => i) as i (i)}
		{@const a = pair ? pair.counts[2 * i] : 0}
		{@const b = pair ? pair.counts[2 * i + 1] : 0}
		{@const v = !pair ? -1 : pair.bits.length ? (pair.bits[2 * i] ? 2 : pair.bits[2 * i + 1] ? 1 : 0) : a > b ? 2 : a === b ? 1 : 0}
		<button
			class="relative h-[30px] rounded-[3px] text-[10px] leading-[30px] tabular-nums transition-colors"
			style:background={v === 2 ? bc.c : v === 1 ? bc.light : v === 0 ? '#E3E8F1' : C.tint}
			style:color={v === 2 ? C.paper : C.ink}
			style:outline={i === chPair ? `2px solid ${C.ink}` : 'none'}
			style:outline-offset="1px"
			title="channel pair {i + 1}: {a} vs {b}{registers[i] ? ' (numbers at the witnesses)' : ''}"
			onclick={() => onchoose(i)}
		>
			{pair ? `${a}·${b}` : ''}
			{#if registers[i]}<span class="absolute top-[2px] right-[3px] h-[3px] w-[3px] rounded-full" style:background={v === 2 ? C.paper : C.ink}></span>{/if}
		</button>
	{/each}
</div>
