import type { PairInfo } from './worker';

/** The witnesses of count channels 2c and 2c + 1 for the selected pair: the middle items that pass both tests. */
export function witnessSets(pair: PairInfo | null, chPair: number, n: number): { a: number[]; b: number[] } | null {
	if (!pair) return null;
	const a: number[] = [];
	const b: number[] = [];
	for (let k = 0; k < n; k++) {
		if (pair.witness[2 * chPair * n + k]) a.push(k);
		if (pair.witness[(2 * chPair + 1) * n + k]) b.push(k);
	}
	return { a, b };
}
