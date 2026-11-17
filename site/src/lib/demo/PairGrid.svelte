<script lang="ts">
	/**
	 * The pair state, read through one count channel pair: a cell for every ordered pair (u, v), rows u and columns v,
	 * coloured by the two bits that channel pair handed on at the loop shown, in the block's colour (cobalt cold, rose
	 * general): full "more" (more witnesses in the first channel), light "same", pale when neither bit is on ("fewer").
	 * Click a cell to select the pair.
	 */
	import { C, blockColour } from './theme';

	let { n, grid, sel, block, onpick }: { n: number; grid: Uint8Array | null; sel: { u: number; v: number } | null; block: string; onpick: (u: number, v: number) => void } = $props();

	let canvas: HTMLCanvasElement | undefined = $state();
	let side = $state(0);
	const cell = $derived(side / n);

	$effect(() => {
		if (!canvas || side <= 0) return;
		const dpr = window.devicePixelRatio || 1;
		const px = Math.round(side * dpr);
		if (canvas.width !== px) {
			canvas.width = px;
			canvas.height = px;
		}
		const g = canvas.getContext('2d')!;
		g.setTransform(dpr, 0, 0, dpr, 0, 0);
		g.fillStyle = C.tint;
		g.fillRect(0, 0, side, side);
		if (grid) {
			const col = ['#E3E8F1', blockColour(block).light, blockColour(block).c];
			for (let u = 0; u < n; u++)
				for (let v = 0; v < n; v++) {
					g.fillStyle = col[grid[u * n + v]];
					g.fillRect(v * cell, u * cell, Math.max(1, cell - (cell > 6 ? 1 : 0)), Math.max(1, cell - (cell > 6 ? 1 : 0)));
				}
		}
		if (sel) {
			// the pair's row and column, faintly; the pair's own cell outlined in ink
			g.fillStyle = C.ink;
			g.globalAlpha = 0.1;
			g.fillRect(0, sel.u * cell, side, cell);
			g.fillRect(sel.v * cell, 0, cell, side);
			g.globalAlpha = 1;
			g.strokeStyle = C.ink;
			g.lineWidth = 2;
			const b = Math.max(cell + 2, 8);
			g.strokeRect(sel.v * cell + cell / 2 - b / 2, sel.u * cell + cell / 2 - b / 2, b, b);
		}
	});

	function pick(e: MouseEvent) {
		const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
		const v = Math.floor(((e.clientX - r.left) / r.width) * n);
		const u = Math.floor(((e.clientY - r.top) / r.height) * n);
		if (u >= 0 && u < n && v >= 0 && v < n) onpick(u, v);
	}
</script>

<div class="relative aspect-square w-full" bind:clientWidth={side}>
	<canvas bind:this={canvas} class="absolute inset-0 h-full w-full cursor-crosshair rounded-[3px]" onclick={pick}></canvas>
</div>
