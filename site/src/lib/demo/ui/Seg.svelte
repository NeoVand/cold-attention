<script lang="ts" generics="T extends string | number">
	/** A choice among a few options: a bordered group of buttons, the chosen one filled in ink. */
	import { HugeiconsIcon, type IconSvgElement } from '@hugeicons/svelte';

	let {
		options,
		value,
		onchoose,
		label,
		disabled = false
	}: {
		options: { id: T; label: string; icon?: IconSvgElement; title?: string }[];
		value: T | null;
		onchoose: (id: T) => void;
		/** what the group chooses, for screen readers */
		label: string;
		disabled?: boolean;
	} = $props();
</script>

<div class="seg" role="group" aria-label={label}>
	{#each options as o (o.id)}
		<button type="button" aria-pressed={value === o.id} title={o.title} {disabled} onclick={() => onchoose(o.id)}>
			{#if o.icon}<HugeiconsIcon icon={o.icon} size={15} strokeWidth={1.6} />{/if}
			{#if o.label}<span>{o.label}</span>{/if}
		</button>
	{/each}
</div>
