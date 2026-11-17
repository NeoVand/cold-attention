/** One async interface over the two engines: engine.ts (CPU, the parity reference) and gpu.ts (WebGPU). */
import { Engine, type Block, type Inputs, type Kind, type ModelConfig, type Params, type State } from './engine';
import { CENSUS_LOOPS, censusLoops, restCap } from './run';

export type Readout = Record<string, { kind: Kind; pred: Int32Array; logits: Float32Array }>;

export interface Runner {
	readonly kind: 'cpu' | 'gpu';
	readonly block: Block;
	readonly model: ModelConfig['model'];
	readonly h: number;
	readonly D: number;
	start(x: Inputs): Promise<State>;
	step(st: State): Promise<State>;
	read(st: State): Promise<Readout>;
	destroy(): void;
}

export function cpuRunner(p: Params, config: ModelConfig): Runner {
	const e = new Engine(p, config);
	return {
		kind: 'cpu',
		block: e.block,
		model: config.model,
		h: e.h,
		D: e.D,
		start: async (x) => e.start(x),
		step: async (st) => e.step(st),
		read: async (st) => e.read(st),
		destroy: () => {}
	};
}

/** A GPU runner when WebGPU is there and asked for, else the CPU one. */
export async function makeRunner(p: Params, config: ModelConfig, want: 'gpu' | 'cpu'): Promise<Runner> {
	if (want === 'gpu') {
		try {
			const { gpuAvailable, GpuEngine } = await import('./gpu');
			const dev = await gpuAvailable();
			if (dev) {
				const g = new GpuEngine(dev, p, config);
				return {
					kind: 'gpu',
					block: g.block,
					model: config.model,
					h: g.h,
					D: g.D,
					start: (x) => g.start(x),
					step: (st) => g.step(st),
					read: (st) => g.read(st),
					destroy: () => g.destroy(),
					get timing() {
						return g.timing;
					}
				} as Runner;
			}
		} catch (err) {
			console.warn('WebGPU unavailable, using the CPU engine', err);
		}
	}
	return cpuRunner(p, config);
}

/** The registered read: the census block after 8 loops; the resting blocks at the first loop whose bits (and numbers) repeat. */
export async function registeredReadAsync(r: Runner, x: Inputs, onLoop?: (st: State) => void) {
	let st = await r.start(x);
	if (r.block === 'census') {
		for (let l = 0, L = censusLoops(r.model, x.n); l < L; l++) {
			st = await r.step(st);
			onLoop?.(st);
		}
	} else {
		const cap = r.block === 'pair_numbers' ? 4 * x.n : x.n;
		for (let l = 1; l <= cap; l++) {
			st = await r.step(st);
			onLoop?.(st);
			if (st.changed === 0) break;
		}
	}
	const out = await r.read(st);
	return { pred: Object.fromEntries(Object.entries(out).map(([k, v]) => [k, v.pred])), loops: st.loop, state: st };
}

export { restCap, CENSUS_LOOPS, censusLoops };
