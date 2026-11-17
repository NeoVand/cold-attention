<script lang="ts">
	/**
	 * Parity, in this browser: every model the page runs, on official instances, against the predictions the Python
	 * release makes (the predictions each model made when it was trained). One worker, models in turn.
	 */
	import { onDestroy } from 'svelte';
	import { asset } from '$app/paths';
	import type { ParityResult } from '#lib/model/paritycore.ts';

	const base = asset('models/index.json').replace(/\/models\/index\.json$/, '');
	let split = $state<'val' | 'test'>('val');
	let limit = $state(8);
	let filter = $state('');
	let engine = $state<'gpu' | 'cpu'>('cpu');
	let rows = $state<(ParityResult & { error?: string })[]>([]);
	let running = $state(false);
	let worker: Worker | null = null;
	onDestroy(() => worker?.terminate());

	async function run() {
		running = true;
		rows = [];
		const index = (await fetch(`${base}/models/index.json`).then((r) => r.json())) as { name: string; task: string }[];
		worker?.terminate();
		worker = new Worker(new URL('../../lib/demo/parityWorker.ts', import.meta.url), { type: 'module' });
		const want = filter.split(',').map((q) => q.trim()).filter(Boolean);
		for (const m of index.filter((q) => !want.length || want.some((w) => q.name.includes(w)))) {
			const r = await new Promise<{ ok: boolean; result?: ParityResult; error?: string }>((res) => {
				worker!.onmessage = (ev) => res(ev.data);
				worker!.postMessage({ base, name: m.name, task: m.task, split, limit, engine });
			});
			rows = [...rows, r.ok ? r.result! : ({ name: m.name, split, start: 0, instances: 0, entries: 0, differ: -1, loopsDiffer: 0, bad: [], seconds: 0, error: r.error } as ParityResult & { error?: string })];
		}
		running = false;
	}
	const total = $derived(rows.reduce((a, r) => ({ entries: a.entries + r.entries, differ: a.differ + Math.max(0, r.differ), loops: a.loops + r.loopsDiffer, inst: a.inst + r.instances }), { entries: 0, differ: 0, loops: 0, inst: 0 }));
</script>

<svelte:head><title>Parity</title></svelte:head>

<div class="mx-auto max-w-[960px] px-4 py-6 sm:px-6">
	<h1 class="text-[22px] font-semibold">Parity with the Python release</h1>
	<p class="text-ink-soft mt-1 max-w-[720px] text-sm">
		Each model runs here, in this browser, on official CLRS-30 instances, and its answers are compared entry by entry with the
		answers the same model gave in Python when it was trained. Read by the registered rule (8 loops; or at rest).
	</p>
	<div class="mt-4 flex flex-wrap items-center gap-4 text-sm">
		<button class="plain" aria-pressed={split === 'val'} onclick={() => (split = 'val')}>val (16 items)</button>
		<button class="plain" aria-pressed={split === 'test'} onclick={() => (split = 'test')}>test (64 items, slow)</button>
		<span class="text-mist">|</span>
		<button class="plain" aria-pressed={engine === 'cpu'} onclick={() => (engine = 'cpu')}>CPU engine (reference)</button>
		<button class="plain" aria-pressed={engine === 'gpu'} onclick={() => (engine = 'gpu')}>WebGPU engine</button>
		<label class="flex items-center gap-2"><span class="hint">instances per model</span><input class="field w-16" type="number" min="1" max="2048" bind:value={limit} /></label>
		<label class="flex items-center gap-2"><span class="hint">models containing</span><input class="field w-56" placeholder="all (or e.g. cold/, mst_prim)" bind:value={filter} /></label>
		<button class="primary" onclick={run} disabled={running}>{running ? 'running…' : 'Run'}</button>
	</div>
	{#if rows.length}
		<p class="mt-4 text-sm" data-testid="parity-total">
			<span class="font-semibold">{total.differ === 0 && total.loops === 0 ? 'Identical' : 'Differences'}</span>: {rows.length} models, {total.inst} instances, {total.entries.toLocaleString()} answer entries, {total.differ} differ, {total.loops} loop counts differ.
		</p>
		<table class="mt-3 w-full text-sm">
			<thead><tr class="label text-left"><th class="py-1">model</th><th class="text-right">instances</th><th class="text-right">entries</th><th class="text-right">differ</th><th class="text-right">loops differ</th><th class="text-right">seconds</th><th class="text-right">engine</th></tr></thead>
			<tbody>
				{#each rows as r (r.name)}
					<tr class="border-t border-[var(--color-edge)]">
						<td class="py-1">{r.name}</td>
						<td class="num text-right">{r.instances}</td>
						<td class="num text-right">{r.entries}</td>
						<td class="num text-right" class:text-rose={r.differ !== 0}>{r.error ? r.error : r.differ}</td>
						<td class="num text-right">{r.loopsDiffer}</td>
						<td class="num text-right">{r.seconds.toFixed(1)}</td>
						<td class="text-right">{r.engine ?? ''}</td>
					</tr>
				{/each}
			</tbody>
		</table>
	{/if}
</div>
