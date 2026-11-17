/** Running the block by its registered read-out rule. */
import type { Engine, Inputs, State } from './engine';

export const CENSUS_LOOPS = 8;

/** The census block's loops for an input of n items: 8, or n for a model read after its own n loops. */
export function censusLoops(model: { read?: { rule: string } } | undefined, n: number): number {
	return model?.read?.rule === 'n' ? n : CENSUS_LOOPS;
}

/** The cap of the resting read: n loops (compare_first) or 4 n (pair_numbers) for n items. */
export function restCap(engine: Engine, n: number): number {
	return engine.block === 'pair_numbers' ? 4 * n : n;
}

/** The registered read: census after 8 loops; resting blocks at the first loop whose bits (and numbers) repeat. */
export function registeredRead(engine: Engine, x: Inputs, onLoop?: (st: State) => void) {
	let st = engine.start(x);
	if (engine.block === 'census') {
		for (let l = 0, L = censusLoops(engine.config.model, x.n); l < L; l++) {
			st = engine.step(st);
			onLoop?.(st);
		}
	} else {
		const cap = restCap(engine, x.n);
		for (let l = 1; l <= cap; l++) {
			st = engine.step(st);
			onLoop?.(st);
			if (st.changed === 0) break;
		}
	}
	const out = engine.read(st);
	return { pred: Object.fromEntries(Object.entries(out).map(([k, v]) => [k, v.pred])), loops: st.loop, state: st };
}
