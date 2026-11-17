<script lang="ts">
	/**
	 * All 30 tasks at a glance: the general block's test score at 64 items, the mean of three seeds,
	 * against FloydNet's; where a cold block exists, its score in cobalt (the general block in rose). Click a task to open it.
	 */
	import { TASKS, FLOYDNET, PROVISIONAL } from './tasks';
	import { C } from './theme';

	let { scores, current, onchoose }: { scores: { task: string; block: string; test: number }[]; current: string; onchoose: (t: string) => void } = $props();

	const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
	const by = $derived.by(() => {
		const m: Record<string, Record<string, number[]>> = {};
		for (const s of scores) if (!PROVISIONAL[s.task]?.includes(s.block)) ((m[s.task] ??= {})[s.block] ??= []).push(s.test);
		return m;
	});
	const census = $derived(TASKS.map((t) => (by[t.id]?.census ? mean(by[t.id].census) : NaN)));
	const avg = $derived(census.every((x) => !Number.isNaN(x)) ? mean(census) : NaN);
	const fAvg = mean(TASKS.map((t) => FLOYDNET[t.id]));
	const fmt = (x: number) => (x >= 99.995 ? '100' : x.toFixed(x >= 99 ? 2 : 1));
</script>

<section>
	<div class="flex flex-wrap items-baseline justify-between gap-2">
		<span class="label">thirty tasks, test at 64 items</span>
		{#if !Number.isNaN(avg)}
			<span class="text-sm">
				average <span class="num font-semibold" style:color={C.rose}>{avg.toFixed(2)}</span>
				<span class="hint">· FloydNet <span class="num">{fAvg.toFixed(2)}</span></span>
			</span>
		{/if}
	</div>
	<div class="mt-2 grid grid-cols-2 gap-[3px] sm:grid-cols-3 lg:grid-cols-5 xl:grid-cols-6">
		{#each TASKS as t, i (t.id)}
			{@const g = census[i]}
			{@const f = FLOYDNET[t.id]}
			{@const cold = by[t.id]?.compare_first ?? by[t.id]?.pair_numbers ?? by[t.id]?.cold}
			{@const coldMean = cold ? mean(cold) : null}
			{@const ahead = g >= f - 1e-9}
			<button
				class="rounded-[3px] px-2 py-1.5 text-left transition-colors"
				style:background={t.id === current ? '#E6EBF4' : C.tint}
				style:outline={t.id === current ? `1.5px solid ${C.ink}` : 'none'}
				onclick={() => onchoose(t.id)}
				title="{t.name}: general block {g.toFixed(2)} (one per task, mean of 3 seeds), FloydNet {f.toFixed(2)}{coldMean !== null ? `, cold block ${coldMean.toFixed(2)}` : ''}"
			>
				<div class="truncate text-[11px] leading-tight text-[var(--color-ink-soft)]">{t.name}</div>
				<div class="mt-0.5 flex items-baseline gap-1.5">
					<span class="num text-[15px] leading-none font-semibold" style:color={C.rose}>{fmt(g)}</span>
					{#if coldMean !== null}<span class="num text-[12px] leading-none font-semibold" style:color={C.cobalt}>{fmt(coldMean)}</span>{/if}
					<span class="num ml-auto text-[11px] leading-none" style:color={ahead ? C.inkSoft : C.ink} style:font-weight={ahead ? 400 : 600}>{fmt(f)}</span>
				</div>
				<div class="relative mt-1 h-[3px] rounded-full" style:background="#E1E6EF">
					<div class="absolute top-0 left-0 h-[3px] rounded-full" style:width="{Math.max(0, (g - 60) / 40) * 100}%" style:background={C.rose}></div>
					<div class="absolute top-[-2px] h-[7px] w-[2px]" style:left="calc({Math.max(0, (f - 60) / 40) * 100}% - 1px)" style:background={C.ink}></div>
				</div>
			</button>
		{/each}
	</div>
	<p class="hint mt-1">
		<span style:color={C.rose} class="font-semibold">Rose</span>: the general block, one per task, the same configuration on all 30 (the bar, from 60 to
		100). <span style:color={C.cobalt} class="font-semibold">Cobalt</span>: the cold block where the task allows it (numbers compared, pair numbers, or read after n loops; on strongly connected components it is below the general block).
		Right, and the tick: FloydNet, 20 to 215 million parameters per task.
	</p>
</section>
