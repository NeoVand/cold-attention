<script lang="ts" module>
	export type Swatch = 'arrow' | 'dashArrow' | 'route' | 'routeDash' | 'fill' | 'light' | 'dashBox' | 'dot' | 'ring' | 'inkDot' | 'inkBox';
</script>

<script lang="ts">
	/** The marks the current view draws, each with a small sample of how it looks. */
	import { C } from '../theme';

	let { items, colour, light }: { items: { kind: Swatch; label: string }[]; colour: string; light: string } = $props();
</script>

<div class="text-ink-soft flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
	{#each items as it (it.label)}
		<span class="inline-flex items-center gap-1.5">
			<svg width="22" height="12" viewBox="0 0 22 12" aria-hidden="true">
				{#if it.kind === 'arrow'}
					<path d="M1,6 H15" stroke={colour} stroke-width="2" stroke-linecap="round" /><path d="M21,6 L14,2.6 L15.6,6 L14,9.4 Z" fill={colour} />
				{:else if it.kind === 'dashArrow'}
					<path d="M1,6 H15" stroke={C.ink} stroke-width="1.5" stroke-dasharray="3 2.5" /><path d="M21,6 L14,2.6 L15.6,6 L14,9.4 Z" fill={C.ink} />
				{:else if it.kind === 'route'}
					<path d="M1,9 Q11,1 21,9" fill="none" stroke={colour} stroke-opacity="0.75" stroke-width="1.8" stroke-linecap="round" />
				{:else if it.kind === 'routeDash'}
					<path d="M1,9 Q11,1 21,9" fill="none" stroke={C.blueGrey} stroke-width="1.6" stroke-dasharray="3 3" />
				{:else if it.kind === 'fill'}
					<rect x="5" y="1" width="12" height="10" rx="1.5" fill={colour} />
				{:else if it.kind === 'light'}
					<rect x="5" y="1" width="12" height="10" rx="1.5" fill={light} />
				{:else if it.kind === 'dashBox'}
					<rect x="5.5" y="1.5" width="11" height="9" rx="1.5" fill="none" stroke={C.ink} stroke-width="1.3" stroke-dasharray="3 2" />
				{:else if it.kind === 'dot'}
					<circle cx="11" cy="6" r="4" fill={colour} />
				{:else if it.kind === 'inkDot'}
					<circle cx="11" cy="6" r="4.5" fill={C.ink} />
				{:else if it.kind === 'inkBox'}
					<rect x="5" y="1" width="12" height="10" rx="2" fill="none" stroke={C.ink} stroke-width="1.8" />
				{:else if it.kind === 'ring'}
					<circle cx="11" cy="6" r="4.5" fill="none" stroke={C.blueGrey} stroke-width="1.5" stroke-dasharray="2.5 2" />
				{/if}
			</svg>
			{it.label}
		</span>
	{/each}
</div>
