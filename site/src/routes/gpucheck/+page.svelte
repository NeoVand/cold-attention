<script lang="ts">
	/** Diagnostic: the CPU and WebGPU engines side by side on one official instance, loop by loop. */
	import { asset } from '$app/paths';
	import { Engine, type ModelConfig } from '#lib/model/engine.ts';
	import { GpuEngine, gpuAvailable } from '#lib/model/gpu.ts';
	import { parseSafetensors } from '#lib/model/safetensors.ts';
	import { decodeArrays, instanceArrays, modelInputs, type SplitJSON } from '#lib/model/inputs.ts';

	const base = asset('models/index.json').replace(/\/models\/index\.json$/, '');
	let name = $state('census/bubble_sort/seed0');
	let which = $state(0);
	let lines = $state<string[]>([]);

	async function go() {
		lines = [];
		const cfg = (await fetch(`${base}/models/${name}/config.json`).then((r) => r.json())) as ModelConfig;
		const p = parseSafetensors(await fetch(`${base}/models/${name}/model.safetensors`).then((r) => r.arrayBuffer()));
		const d = decodeArrays((await fetch(`${base}/instances/${cfg.task}/test.json`).then((r) => r.json())) as SplitJSON);
		const { n, arrays } = instanceArrays(d, which);
		const x = modelInputs(cfg.model, n, arrays);
		const cpu = new Engine(p, cfg);
		const dev = await gpuAvailable();
		if (!dev) return (lines = ['no WebGPU']);
		const gpu = new GpuEngine(dev, p, cfg);
		let a = cpu.start(x);
		let b = await gpu.start(x);
		for (let l = 1; l <= 8; l++) {
			const za = cpu.state(a);
			a = cpu.step(a);
			b = await gpu.step(b);
			let maxd = 0, maxv = 0, flipsA = 0, flipsB = 0, cnt = 0;
			for (let i = 0; i < a.s.length; i++) {
				maxd = Math.max(maxd, Math.abs(a.s[i] - b.s[i]));
				maxv = Math.max(maxv, Math.abs(a.s[i]));
			}
			if (a.trace && b.trace) {
				for (let i = 0; i < a.trace.testA.length; i++) {
					if (a.trace.testA[i] !== b.trace.testA[i]) flipsA++;
					if (a.trace.testB[i] !== b.trace.testB[i]) flipsB++;
					if (a.trace.counts[i] !== b.trace.counts[i]) cnt++;
				}
			}
			void za;
			const pa = cpu.read(a);
			const pb = await gpu.read(b);
			let pd = 0;
			for (const k of Object.keys(pa)) for (let i = 0; i < pa[k].pred.length; i++) if (pa[k].pred[i] !== pb[k].pred[i]) pd++;
			lines = [...lines, `loop ${l}: state max |cpu - gpu| ${maxd.toExponential(2)} (max |state| ${maxv.toFixed(2)}), left-test flips ${flipsA}, right-test flips ${flipsB}, counts differ ${cnt}, predictions differ ${pd}`];
		}
		gpu.destroy();
	}
</script>

<div class="mx-auto max-w-[960px] px-4 py-6 text-sm">
	<h1 class="text-lg font-semibold">CPU and WebGPU engines, loop by loop</h1>
	<div class="mt-3 flex gap-3">
		<input class="field w-80" bind:value={name} />
		<input class="field w-20" type="number" bind:value={which} />
		<button class="primary" onclick={go}>Run</button>
	</div>
	<pre class="mt-4 text-xs" data-testid="out">{lines.join('\n')}</pre>
</div>
