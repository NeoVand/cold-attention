<script lang="ts">
	/**
	 * The loops, left to right. A dot per loop: how right the answer read after that loop is (top = all right). For the
	 * resting blocks a bar under it: how many bits changed in that loop (log scale); the block is read when none do.
	 * Click a loop to show it.
	 */
	import { C, blockColour } from './theme';

	let {
		loops,
		shown,
		resting,
		cap,
		block,
		onshow
	}: {
		loops: { loop: number; changed: number; acc: number }[];
		shown: number;
		resting: boolean;
		cap: number;
		block: string;
		onshow: (l: number) => void;
	} = $props();

	/** census: its 8 loops; resting blocks: the loops run so far, with room (the cap, n or 4 n, is far away) */
	const slots = $derived(resting ? Math.max(loops.length + 1, 13) : Math.max(cap + 1, loops.length));
	const maxCh = $derived(Math.max(1, ...loops.map((l) => l.changed)));
	const W = 1000;
	const H = 92;
	const sw = $derived((W - 20) / slots);
	const xc = (l: number) => 10 + (l + 0.5) * sw;
</script>

<svg viewBox="0 0 {W} {H}" class="w-full select-none" role="img" aria-label="loops">
	<line x1="10" x2={W - 10} y1="62" y2="62" stroke={C.edge} />
	{#each loops as l (l.loop)}
		<g role="button" tabindex="-1" class="cursor-pointer" onclick={() => onshow(l.loop)} onkeydown={(e) => e.key === 'Enter' && onshow(l.loop)}>
			<rect x={xc(l.loop) - sw / 2} y="0" width={sw} height={H} fill={l.loop === shown ? C.tint : 'transparent'} />
			{#if resting && l.loop > 0}
				{@const h = l.changed > 0 ? 6 + (Math.log1p(l.changed) / Math.log1p(maxCh)) * 22 : 0}
				<rect x={xc(l.loop) - Math.min(8, sw * 0.3)} y={90 - h} width={Math.min(16, sw * 0.6)} height={h} fill={C.mist} rx="1" />
				{#if l.changed === 0}<text x={xc(l.loop)} y="86" text-anchor="middle" font-size="12" fill={C.inkSoft}>rest</text>{/if}
			{/if}
			<circle cx={xc(l.loop)} cy={8 + (1 - l.acc) * 50} r={l.loop === shown ? 5.5 : 4} fill={l.acc === 1 ? blockColour(block).c : C.paper} stroke={blockColour(block).c} stroke-width="1.8" />
			{#if sw > 16}<text x={xc(l.loop)} y="76" text-anchor="middle" font-size="11" fill={l.loop === shown ? C.ink : C.blueGrey} class="num">{l.loop}</text>{/if}
		</g>
	{/each}
</svg>
