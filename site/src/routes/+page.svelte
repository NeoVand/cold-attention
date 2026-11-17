<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { asset } from '$app/paths';
	import { Demo, type Instance } from '#lib/demo/demo.svelte.ts';
	import { TASKS, TASK, GROUPS, BLOCK_LABEL, BLOCK_NOTE, RECODE } from '#lib/demo/tasks.ts';
	import RecodePanel from '#lib/demo/RecodePanel.svelte';
	import { instanceArrays, type Arrays } from '#lib/model/inputs.ts';
	import { sample, itemsFor, minLength } from '#lib/clrs/samplers.ts';
	import { makeRng } from '#lib/clrs/rng.ts';
	import { toPadded, type Task } from '#lib/clrs/instances.ts';
	import { scoreInstance, mean } from '#lib/demo/score.ts';
	import { witnessSets } from '#lib/demo/witness.ts';
	import GraphView from '#lib/demo/views/GraphView.svelte';
	import ArrayView from '#lib/demo/views/ArrayView.svelte';
	import TableView from '#lib/demo/views/TableView.svelte';
	import IntervalView from '#lib/demo/views/IntervalView.svelte';
	import StringView from '#lib/demo/views/StringView.svelte';
	import GeometryView from '#lib/demo/views/GeometryView.svelte';
	import PairGrid from '#lib/demo/PairGrid.svelte';
	import ChannelStrip from '#lib/demo/ChannelStrip.svelte';
	import LoopStrip from '#lib/demo/LoopStrip.svelte';
	import ScoreGrid from '#lib/demo/ScoreGrid.svelte';
	import { C, blockColour, isGeneral } from '#lib/demo/theme.ts';
	import Seg from '#lib/demo/ui/Seg.svelte';
	import IconButton from '#lib/demo/ui/IconButton.svelte';
	import Legend, { type Swatch } from '#lib/demo/ui/Legend.svelte';
	import { HugeiconsIcon } from '@hugeicons/svelte';
	import {
		ArrowLeft01Icon,
		ArrowRight01Icon,
		ChipIcon,
		CpuIcon,
		CubeIcon,
		DiceIcon,
		File02Icon,
		GithubIcon,
		Layers01Icon,
		Cancel01Icon,
		PauseIcon,
		PlayIcon,
		PreviousIcon,
		NextIcon
	} from '@hugeicons/core-free-icons';

	const base = asset('models/index.json').replace(/\/models\/index\.json$/, '');
	const demo = new Demo(base);
	let freshSeed = $state(1);
	let playing = $state(false);

	onMount(() => {
		demo.init();
	});
	onDestroy(() => demo.destroy());

	function draw() {
		const task = demo.task as Task;
		const seed = freshSeed++;
		demo.source = 'fresh';
		demo.loadInstance((): Instance => {
			const inst = sample(task, demo.size, makeRng(seed));
			const { n, arrays } = instanceArrays(toPadded([inst], inst.n) as Arrays, 0);
			return { task, n, arrays, origin: `fresh instance, ${n} items (draw ${seed})` };
		});
	}
	function official(split: 'val' | 'test', which = 0) {
		demo.source = 'official';
		demo.split = split;
		demo.which = which;
		demo.loadInstance();
	}
	function step(d: number) {
		demo.which = Math.max(0, Math.min(demo.splitCount - 1, demo.which + d));
		demo.loadInstance();
	}

	const view = $derived(demo.instance ? TASK[demo.instance.task].family : null);
	const pred = $derived(demo.current?.pred ?? null);
	const scores = $derived(demo.instance && pred && demo.config && demo.config.task === demo.instance.task ? scoreInstance(demo.config.outputs, pred, demo.instance.arrays, demo.instance.n) : null);
	/** the recoding test: live tasks, or any listed task with ?recode in the address */
	const recode = $derived.by(() => {
		const r = RECODE[demo.task];
		if (!r) return null;
		const ask = typeof location !== 'undefined' && new URLSearchParams(location.search).has('recode');
		return r.live || ask ? r : null;
	});
	/** the general block (one per task, or the shared one) is read after a fixed count of loops; the cold blocks run until they rest */
	const general = $derived(isGeneral(demo.block));
	const bc = $derived(blockColour(demo.block));
	const resting = $derived(!general);
	const cap = $derived(
		!demo.instance
			? 8
			: general
				? demo.config?.model.read?.rule === 'n'
					? demo.instance.n
					: 8
				: demo.block === 'pair_numbers'
					? 4 * demo.instance.n
					: demo.instance.n
	);
	const loopRows = $derived(
		demo.loops.map((l) => ({
			loop: l.loop,
			changed: l.changed,
			acc: demo.instance && demo.config && demo.config.task === demo.instance.task ? mean(scoreInstance(demo.config.outputs, l.pred, demo.instance.arrays, demo.instance.n)) : 0
		}))
	);
	const registers = $derived.by(() => {
		const r = Array(32).fill(false);
		if (demo.block === 'compare_first') for (let i = 16; i < 32; i++) r[i] = true;
		if (demo.block === 'pair_numbers') for (let i = 16; i < 32; i++) r[i] = true;
		return r;
	});
	const wit = $derived(demo.instance && demo.sel ? witnessSets(demo.pair, demo.chPair, demo.instance.n) : null);
	const sizeRange = $derived.by(() => {
		const [d, lo, hi] = demo.info.size;
		return [d, Math.max(lo, minLength(demo.task as Task)), hi] as const;
	});
	const freshItems = $derived(itemsFor(demo.task as Task, demo.size));
	const timing = $derived(demo.loops.length > 1 ? demo.loops.slice(1).reduce((a, l) => a + l.ms, 0) / (demo.loops.length - 1) : 0);

	// play: step the shown loop through the loops that have arrived
	$effect(() => {
		if (!playing) return;
		const id = setInterval(() => {
			const last = demo.loops.at(-1)?.loop ?? 0;
			if (demo.shown >= last) {
				if (!demo.running) playing = false;
				return;
			}
			demo.show(demo.shown + 1);
		}, 650);
		return () => clearInterval(id);
	});
	/** play the loops from the one shown (from the first when the last is shown); again to pause */
	function play() {
		if (playing) {
			playing = false;
			return;
		}
		if (demo.shown >= (demo.loops.at(-1)?.loop ?? 0)) demo.show(0);
		playing = true;
	}
	function stepLoop(d: number) {
		playing = false;
		const last = demo.loops.at(-1)?.loop ?? 0;
		demo.show(Math.max(0, Math.min(last, demo.shown + d)));
	}
	/** the instance source: an official split, or a fresh draw */
	const instanceKind = $derived(demo.source === 'fresh' ? 'fresh' : demo.split);
	function chooseInstance(k: 'val' | 'test' | 'fresh') {
		if (k === 'fresh') draw();
		else official(k);
	}
	const sorts = $derived(['insertion_sort', 'bubble_sort', 'heapsort', 'quicksort'].includes(demo.task));
	/** the marks the view draws, for the legend under it */
	const legend = $derived.by((): { kind: Swatch; label: string }[] => {
		const who = general ? 'the general block' : 'the cold block';
		const witnesses: { kind: Swatch; label: string }[] = demo.sel
			? view === 'graph'
				? [
						{ kind: 'inkDot', label: 'the pair u, v' },
						{ kind: 'route', label: 'witnesses, first channel' },
						{ kind: 'ring', label: 'second channel' }
					]
				: [
						{ kind: 'inkBox', label: 'the pair u, v' },
						{ kind: 'dot', label: 'witnesses, first channel' },
						{ kind: 'ring', label: 'second channel' }
					]
			: [];
		if (view === 'graph') {
			const edgeTask = ['bridges', 'mst_kruskal', 'articulation_points'].includes(demo.task);
			return [
				{ kind: edgeTask ? 'fill' : 'arrow', label: `${who}'s answer` },
				{ kind: edgeTask ? 'dashBox' : 'dashArrow', label: 'the true answer, where it differs' },
				...witnesses
			];
		}
		if (view === 'array' && sorts)
			return [
				{ kind: 'fill', label: `in the slot ${who}'s pointers give` },
				{ kind: 'light', label: 'out of place' },
				{ kind: 'dashBox', label: 'the true bar in that slot' },
				...witnesses
			];
		if (demo.task === 'find_maximum_subarray_kadane')
			return [
				{ kind: 'fill', label: `in the run ${who} chooses, rightly` },
				{ kind: 'light', label: 'chosen, wrongly' },
				{ kind: 'dashBox', label: 'missed: in the true run' },
				...witnesses
			];
		if (view === 'table') return [{ kind: 'light', label: `an entry ${who} gets wrong (hover a cell for the true one)` }];
		if (view === 'string')
			return [
				{ kind: 'light', label: `where ${who} puts the needle` },
				{ kind: 'dashBox', label: 'the true first match, where it differs' },
				...witnesses
			];
		if (demo.task === 'segments_intersect') return witnesses;
		return [
			{ kind: 'fill', label: `chosen by ${who}, rightly` },
			{ kind: 'light', label: 'chosen, wrongly' },
			{ kind: 'dashBox', label: 'missed: in the true answer' },
			...witnesses
		];
	});
	const pct = (x: number) => `${(100 * x).toFixed(x === 1 ? 0 : 1)}%`;
</script>

<svelte:window onkeydown={(e) => e.key === 'Escape' && demo.clearSel()} />

<svelte:head>
	<title>Attention at absolute zero</title>
	<meta name="description" content="Cold attention: a looped attention block that counts and compares witnesses at zero temperature, running in the browser on all 30 CLRS-30 algorithms." />
</svelte:head>

<div class="mx-auto max-w-[1440px] px-4 pb-10 sm:px-6">
	<header class="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 pt-5 pb-4">
		<div>
			<h1 class="text-[22px] font-semibold tracking-tight">Attention at absolute zero</h1>
			<p class="text-ink-soft mt-0.5 max-w-[820px] text-sm">
				<strong class="font-semibold text-ink">Cold attention</strong>: a looped block of about 150,000 parameters, trained on
				inputs of at most 16 items, running here in your browser on all 30 algorithms of CLRS-30, trained without hints. One general
				block shared by all 30 tasks averages 97.20% at 64 items over three seeds, against 96.64% for FloydNet (one model per
				task); one general block per task, 97.04%; each task in the setting the theory prescribes, 97.98% on the benchmark's
				standard inputs.{' '}<span class="hidden sm:inline">Each loop asks two
				yes-or-no questions of every middle item; an item that passes both is a <em>witness</em> for the pair. At zero
				temperature a sum over witnesses keeps two exact numbers, the best witness and how many there are; the layer
				computes them and passes on only comparisons. Below: Bellman-Ford on a 64-node test graph, four times the
				training size, solved by the cold block with pair numbers. Click two nodes to see a pair's witnesses.</span>
			</p>
		</div>
		<nav class="flex flex-wrap gap-x-4 gap-y-1 text-sm" aria-label="links">
			<a class="ilink" href="https://arxiv.org/abs/2611.11937" title="arXiv:2611.11937"><HugeiconsIcon icon={File02Icon} size={16} strokeWidth={1.6} />Paper</a>
			<a class="ilink" href="https://github.com/NeoVand/cold-attention"><HugeiconsIcon icon={GithubIcon} size={16} strokeWidth={1.6} />Code</a>
			<a class="ilink" href="https://huggingface.co/mohsenvand/cold-attention-clrs30"><HugeiconsIcon icon={CubeIcon} size={16} strokeWidth={1.6} />Models</a>
			<a class="ilink" href="https://huggingface.co/mohsenvand/cold-attention-clrs30-shared"><HugeiconsIcon icon={Layers01Icon} size={16} strokeWidth={1.6} />Shared block</a>
		</nav>
	</header>

	<div class="grid grid-cols-1 gap-6 lg:grid-cols-[210px_minmax(0,1fr)_300px]">
		<!-- controls -->
		<aside class="flex flex-col gap-5">
			<select class="field lg:hidden" value={demo.task} onchange={(e) => demo.choose((e.currentTarget as HTMLSelectElement).value)} aria-label="task">
				{#each GROUPS as g (g)}
					<optgroup label={g}>
						{#each TASKS.filter((t) => t.group === g) as t (t.id)}<option value={t.id}>{t.name}</option>{/each}
					</optgroup>
				{/each}
			</select>
			<nav class="hidden flex-col gap-3 lg:flex" aria-label="tasks">
				{#each GROUPS as g (g)}
					<div>
						<div class="label mb-1">{g}</div>
						<div class="flex flex-col">
							{#each TASKS.filter((t) => t.group === g) as t (t.id)}
								<button class="navitem" aria-current={demo.task === t.id} onclick={() => demo.choose(t.id)}>{t.name}</button>
							{/each}
						</div>
					</div>
				{/each}
			</nav>
		</aside>

		<!-- the task -->
		<main class="flex min-w-0 flex-col">
			<div class="order-1 flex flex-wrap items-baseline justify-between gap-3 lg:order-none">
				<div>
					<h2 class="text-lg font-semibold">{demo.info.name}</h2>
					<p class="text-ink-soft text-sm">{demo.info.asks}</p>
				</div>
				<div class="flex flex-wrap items-center gap-2">
					<Seg label="block" options={demo.blocks.map((b) => ({ id: b, label: BLOCK_LABEL[b] }))} value={demo.block} onchoose={(b) => demo.setBlock(b)} />
					{#if demo.seeds.length > 1}
						<Seg label="seed" options={demo.seeds.map((s) => ({ id: s, label: `seed ${s}` }))} value={demo.seed} onchoose={(s) => demo.setSeed(s)} />
					{/if}
				</div>
			</div>
			<p class="hint order-6 mt-1 lg:order-none">{BLOCK_LABEL[demo.block]}: {BLOCK_NOTE[demo.block]}.{#if demo.model}&nbsp;Official test score {demo.model.scores.test.toFixed(2)}%, {demo.model.parameters.toLocaleString()} parameters.{/if}</p>

			<div class="order-5 mt-3 flex flex-wrap items-center gap-x-3 gap-y-2 text-sm lg:order-none">
				<Seg
					label="instance"
					options={[
						{ id: 'val' as const, label: 'Official, 16 items', title: 'the official validation inputs, at the training size' },
						{ id: 'test' as const, label: 'Official test, 64', title: 'the official test inputs, four times the training size' },
						{ id: 'fresh' as const, label: 'Fresh', title: 'a new input from the benchmark\'s sampler, at the size you set' }
					]}
					value={instanceKind}
					onchoose={chooseInstance}
				/>
				{#if demo.source === 'official'}
					<span class="flex items-center gap-1.5">
						<IconButton icon={ArrowLeft01Icon} label="previous instance" onclick={() => step(-1)} disabled={demo.which === 0} />
						<span class="num text-ink-soft min-w-[52px] text-center">{demo.which + 1} / {demo.splitCount}</span>
						<IconButton icon={ArrowRight01Icon} label="next instance" onclick={() => step(1)} disabled={demo.which >= demo.splitCount - 1} />
					</span>
				{:else}
					<label class="flex items-center gap-2">
						<input type="range" min={sizeRange[1]} max={sizeRange[2]} bind:value={demo.size} class="w-28" disabled={sizeRange[1] === sizeRange[2]} aria-label="items in a fresh instance" />
						<span class="num text-ink-soft w-[64px]">{freshItems} items</span>
					</label>
					<IconButton icon={DiceIcon} label="Draw another" show onclick={draw} />
				{/if}
			</div>

			<div class="bg-paper relative order-2 mt-3 aspect-[10/7] w-full overflow-hidden rounded-[4px] border border-[var(--color-edge)] lg:order-none">
				{#if demo.instance}
					{@const p = { task: demo.instance.task, block: demo.block, n: demo.instance.n, arrays: demo.instance.arrays, pred, sel: demo.sel, pending: demo.pending, pair: demo.pair, chPair: demo.chPair, onitem: (k: number) => demo.pickItem(k), onclear: () => demo.clearSel() }}
					{#key demo.instance}
						{#if view === 'graph'}<GraphView {...p} />
						{:else if view === 'array'}<ArrayView {...p} />
						{:else if view === 'table'}<TableView task={p.task} block={p.block} n={p.n} arrays={p.arrays} {pred} sel={p.sel} onpick={(u, v) => demo.select(u, v)} />
						{:else if view === 'interval'}<IntervalView {...p} />
						{:else if view === 'string'}<StringView {...p} />
						{:else}<GeometryView {...p} />{/if}
					{/key}
				{/if}
				{#if demo.loading}<div class="hint absolute top-2 left-3">loading {demo.loading}…</div>{/if}
				{#if demo.pending !== null || demo.sel}
					<div class="text-ink absolute top-2 left-2 flex items-center gap-1.5 rounded-[6px] border border-[var(--color-edge)] bg-white/90 py-0.5 pr-0.5 pl-2.5 text-xs">
						<span class="num">
							{#if demo.sel}pair u = {demo.sel.u}, v = {demo.sel.v}{:else}u = {demo.pending}; click a second item for v{/if}
						</span>
						<button class="plain flex h-5 w-5 items-center justify-center rounded-[4px] hover:bg-[var(--color-tint)]" aria-label="clear the pair" title="clear (Esc)" onclick={() => demo.clearSel()}>
							<HugeiconsIcon icon={Cancel01Icon} size={13} strokeWidth={1.8} />
						</button>
					</div>
				{/if}
			</div>
			<div class="order-3 mt-1.5 flex flex-wrap items-center justify-between gap-x-4 gap-y-1 lg:order-none">
				<Legend items={legend} colour={bc.c} light={bc.light} />
				<span class="hint">{demo.instance?.origin ?? ''}</span>
			</div>

			<div class="order-4 mt-4 lg:order-none">
				<div class="mb-2 flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
					<span class="label">loops {#if resting}(until the bits stop changing, at most {cap}){:else}(read after {cap}){/if}</span>
					<span class="flex items-center gap-1.5 text-sm">
						{#if timing}<span class="hint num mr-1.5">{timing.toFixed(0)} ms a loop</span>{/if}
						<IconButton icon={PreviousIcon} label="previous loop" onclick={() => stepLoop(-1)} disabled={demo.shown <= 0} />
						<IconButton icon={playing ? PauseIcon : PlayIcon} label={playing ? 'pause' : 'play the loops'} onclick={play} disabled={!demo.loops.length} />
						<IconButton icon={NextIcon} label="next loop" onclick={() => stepLoop(1)} disabled={demo.shown >= (demo.loops.at(-1)?.loop ?? 0)} />
						<span class="ml-1.5">
							<Seg
								label="engine"
								options={[
									{ id: 'gpu' as const, label: 'GPU', icon: ChipIcon, title: 'WebGPU: fast' },
									{ id: 'cpu' as const, label: 'CPU', icon: CpuIcon, title: 'the CPU engine: the parity reference' }
								]}
								value={demo.engineUsed === 'cpu' ? 'cpu' : demo.engineUsed === 'gpu' ? 'gpu' : null}
								onchoose={(e) => demo.setEngine(e)}
							/>
						</span>
					</span>
				</div>
				<LoopStrip loops={loopRows} shown={demo.shown} {resting} {cap} block={demo.block} onshow={(l) => demo.show(l)} />
				<p class="hint">Dots: share of the answer right when read after that loop.{#if resting}{' '}Bars: bits that changed in that loop; the answer is read at the first loop where none do.{/if}</p>
			</div>
			{#if recode && demo.instance && demo.instance.task === demo.task}
				<div class="order-7 lg:order-none">
					<RecodePanel base={demo.base} task={demo.task} n={demo.instance.n} arrays={demo.instance.arrays} engine={demo.engineUsed === 'cpu' ? 'cpu' : 'gpu'} models={recode.models} note={recode.note} />
				</div>
			{/if}
		</main>

		<!-- the instrument -->
		<aside class="flex flex-col gap-5">
			<section>
				<div class="label mb-1">answer</div>
				{#if scores}
					<div class="flex items-baseline gap-2">
						<span class="num text-[28px] leading-none font-semibold" style:color={mean(scores) === 1 ? bc.c : C.ink}>{pct(mean(scores))}</span>
						<span class="hint">right, read after loop {demo.current?.loop}</span>
					</div>
					{#each scores as s (s.name)}
						<div class="hint num mt-1">{s.name}: {s.right} of {s.total} entries right{s.kind.endsWith('mask') ? ` · F1 ${s.score.toFixed(3)}` : ''}</div>
					{/each}
					<div class="hint mt-1">
						{#if demo.running}running…{:else if resting}{demo.rested ? `at rest after ${demo.loops.at(-1)?.loop} loops` : `stopped at the cap, ${cap} loops`}{:else}read after {cap} loops{/if}
					</div>
				{:else}
					<div class="hint">…</div>
				{/if}
			</section>

			<section>
				<div class="flex items-baseline justify-between">
					<span class="label">pair state</span>
					<span class="hint num">channel pair {demo.chPair + 1}, loop {demo.shown}</span>
				</div>
				{#if demo.instance}
					<div class="mt-1"><PairGrid n={demo.instance.n} grid={demo.grid} sel={demo.sel} block={demo.block} onpick={(u, v) => demo.select(u, v)} /></div>
				{/if}
				<p class="hint mt-1 flex flex-wrap items-center justify-between gap-x-3 gap-y-0.5">
					<span class="flex items-center gap-3">
						<span class="flex items-center gap-1"><span class="inline-block h-2.5 w-2.5 rounded-[2px]" style:background={bc.c}></span>more</span>
						<span class="flex items-center gap-1"><span class="inline-block h-2.5 w-2.5 rounded-[2px]" style:background={bc.light}></span>same</span>
						<span class="flex items-center gap-1"><span class="inline-block h-2.5 w-2.5 rounded-[2px]" style:background="#E3E8F1"></span>fewer</span>
					</span>
					<span>row u, column v · click a cell to select the pair</span>
				</p>
			</section>

			<section>
				<div class="flex items-baseline justify-between">
					<span class="label">pair</span>
					<span class="num text-sm">{demo.sel ? `u = ${demo.sel.u}, v = ${demo.sel.v}` : 'click a cell or two items'}</span>
				</div>
				<div class="mt-2"><ChannelStrip pair={demo.pair} chPair={demo.chPair} {registers} block={demo.block} onchoose={(c) => demo.setChPair(c)} /></div>
				{#if demo.sel && wit}
					<p class="mt-2 text-sm">
						Channel pair {demo.chPair + 1}: <span style:color={bc.c} class="font-semibold">{wit.a.length}</span> witness{wit.a.length === 1 ? '' : 'es'} pass the
						first channel's tests, <span style:color={C.blueGrey} class="font-semibold">{wit.b.length}</span> the second's{#if registers[demo.chPair]};
						this pair compares the inputs' numbers at its witnesses{/if}.
					</p>
				{:else}
					<p class="hint mt-2">Each cell is one channel pair: its two witness counts for the selected pair, coloured by the comparison it hands on.</p>
				{/if}
			</section>
		</aside>
	</div>

	<div class="mt-8">
		<ScoreGrid scores={demo.scores} current={demo.task} onchoose={(t) => { demo.choose(t); window.scrollTo({ top: 0, behavior: 'smooth' }); }} />
	</div>
</div>
