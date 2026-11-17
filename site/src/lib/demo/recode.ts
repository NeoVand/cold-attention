/**
 * Recoding positions: one strictly increasing function applied to every node's position. The order of the nodes is
 * unchanged, so the true depth-first parents are unchanged; a block that reads positions only through comparisons (the
 * compare-first block, positions in its register) gives the same answer, a block that embeds them (the cold block)
 * need not.
 */
import type { Arrays } from '../model/inputs';

export type Recoding = 'half' | 'sqrt' | 'square' | 'random';

export const RECODINGS: { id: Recoding; label: string; title: string }[] = [
	{ id: 'half', label: 'pos / 2', title: 'every position halved' },
	{ id: 'sqrt', label: '√pos', title: 'the square root of every position' },
	{ id: 'square', label: 'pos²', title: 'every position squared' },
	{ id: 'random', label: 'random increasing', title: 'n uniform numbers, sorted, the k-th smallest to the node whose position is k-th smallest' }
];

/** The instance's arrays with every position mapped by the recoding (float64 maths, stored as float32 like the files). */
export function recodePositions(arrays: Arrays, n: number, r: Recoding, rand: () => number = Math.random): Arrays {
	const pos = Array.from(arrays.pos.data.subarray(0, n), Number);
	let out: number[];
	if (r === 'random') {
		const draws = Array.from({ length: n }, rand).sort((a, b) => a - b);
		const order = pos.map((p, i) => [p, i] as const).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
		out = new Array(n);
		order.forEach(([, i], k) => (out[i] = draws[k]));
	} else out = pos.map((p) => (r === 'half' ? p / 2 : r === 'sqrt' ? Math.sqrt(p) : p * p));
	const data = new Float32Array(arrays.pos.data.length);
	data.set(out);
	return { ...arrays, pos: { dtype: 'f32', shape: arrays.pos.shape, data } };
}

/** Is the recoded order the original order (ties kept)? A check the page shows next to the result. */
export function sameOrder(a: ArrayLike<number>, b: ArrayLike<number>, n: number): boolean {
	for (let i = 0; i < n; i++)
		for (let j = 0; j < n; j++) if (Math.sign(a[i] - a[j]) !== Math.sign(b[i] - b[j])) return false;
	return true;
}
