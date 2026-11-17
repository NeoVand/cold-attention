<script lang="ts">
	/**
	 * The recoding test, live: one increasing function of every node's position (the true answer does not move), the same
	 * graph through a block that reads positions only through comparisons and through one that embeds them, and how many
	 * of each block's parents moved against its own run on the graph as given.
	 */
	import { onDestroy } from 'svelte';
	import type { Arrays } from '../model/inputs';
	import GraphView from './views/GraphView.svelte';
	import { RECODINGS, recodePositions, type Recoding } from './recode';
	import type { RecodeResult } from './recodeWorker';
	import { C, blockColour } from './theme';
	import Seg from './ui/Seg.svelte';

	let {
		base,
		task,
		n,
		arrays,
		engine,
		models,
		note
	}: {
		base: string;
		task: string;
		n: number;
		arrays: Arrays;
		engine: 'gpu' | 'cpu';
		models: { name: string; label: string; note: string }[];
		/** what the full measurement found (shown under the live counts) */
		note?: string;
	} = $props();

	let worker: Worker | null = null;
	let id = 0;
	let running = $state<string | null>(null);
	let error = $state<string | null>(null);
	/** the last run, kept with the instance it was made on (a new instance hides it) */
	let last = $state.raw<{ of: Arrays; chosen: Recoding; recoded: Arrays; results: RecodeResult[] | null } | null>(null);
	const cur = $derived(last && last.of === arrays ? last : null);
	const chosen = $derived(cur?.chosen ?? null);
	const results = $derived(cur?.results ?? null);
	const recoded = $derived(cur?.recoded ?? null);
	onDestroy(() => worker?.terminate());

	function run(r: Recoding) {
		error = null;
		const of = arrays;
		const rec = recodePositions(arrays, n, r);
		last = { of, chosen: r, recoded: rec, results: null };
		worker ??= new Worker(new URL('./recodeWorker.ts', import.meta.url), { type: 'module' });
		const my = ++id;
		running = 'loading the two models';
		worker.onmessage = (ev) => {
			const m = ev.data;
			if (m.id !== my) return;
			if (m.progress) running = m.progress;
			else if (m.error) {
				error = m.error;
				running = null;
			} else {
				last = { of, chosen: r, recoded: rec, results: m.results };
				running = null;
			}
		};
		worker.postMessage({ id: my, base, names: models.map((q) => q.name), n, original: $state.snapshot(of), recoded: rec, engine });
	}

	const out = (r: RecodeResult) => Object.keys(r.original)[0];
	const moved = (r: RecodeResult) => {
		const k = out(r);
		let c = 0;
		for (let i = 0; i < n; i++) if (r.original[k][i] !== r.recoded[k][i]) c++;
		return c;
	};
	const right = (r: RecodeResult) => {
		const k = out(r);
		let c = 0;
		for (let i = 0; i < n; i++) if (r.recoded[k][i] === arrays[k].data[i]) c++;
		return c;
	};
	const label = (name: string) => models.find((q) => q.name === name)!;
</script>

<section class="mt-5">
	<div class="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
		<span class="label">recode the positions</span>
		<Seg label="recoding" options={RECODINGS.map((r) => ({ id: r.id, label: r.label, title: r.title }))} value={chosen} onchoose={(id) => run(id)} disabled={!!running} />
	</div>
	<p class="hint mt-1">
		One increasing function of every node's position: the nodes keep their order, so the true parents stay where they are. Each
		block runs on the graph as given and on the recoded one; the count is the parents that moved.
	</p>
	{#if running}<p class="hint mt-2">{running}…</p>{/if}
	{#if error}<p class="mt-2 text-sm font-semibold" style:color={C.ink}>{error}</p>{/if}
	{#if results && recoded}
		<div class="mt-2 grid grid-cols-1 gap-4 sm:grid-cols-2">
			{#each results as r (r.name)}
				{@const m = moved(r)}
				<div>
					<div class="flex items-baseline justify-between gap-2">
						<span class="text-sm font-semibold">{label(r.name).label}</span>
						<span class="num text-sm"><span class="font-semibold" style:color={blockColour(r.block).c}>{m}</span> of {n} parents moved</span>
					</div>
					<div class="hint">{label(r.name).note} · right after recoding: {right(r)} of {n}</div>
					<div class="mt-1 aspect-[10/7] w-full rounded-[3px] border" style:border-color={C.edge}>
						<GraphView {task} block={r.block} {n} arrays={recoded} pred={r.recoded} sel={null} pending={null} pair={null} chPair={0} onitem={() => {}} onclear={() => {}} />
					</div>
				</div>
			{/each}
		</div>
		<p class="hint mt-1">Each graph: the block's answer on the recoded positions; dashed ink where it differs from the true answer.</p>
	{/if}
	{#if note}<p class="hint mt-2">{note}</p>{/if}
</section>
