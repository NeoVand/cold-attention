<script lang="ts">
	/** An action: an outlined button with an icon; the label shows beside it when `show`, and as its tooltip always. */
	import { HugeiconsIcon, type IconSvgElement } from '@hugeicons/svelte';

	let {
		icon,
		label,
		onclick,
		disabled = false,
		show = false
	}: {
		icon: IconSvgElement;
		label: string;
		onclick: () => void;
		disabled?: boolean;
		show?: boolean;
	} = $props();
</script>

<button type="button" class="ibtn" title={label} aria-label={label} {disabled} {onclick}>
	<HugeiconsIcon {icon} size={16} strokeWidth={1.6} />
	{#if show}<span>{label}</span>{/if}
</button>
