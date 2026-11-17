import { describe, expect, test } from 'vitest';
import type { Arrays } from '../model/inputs';
import { recodePositions, sameOrder, RECODINGS } from './recode';

const n = 16;
const arrays: Arrays = {
	pos: { dtype: 'f32', shape: [n], data: Float32Array.from({ length: n }, (_, i) => i / n) },
	pi: { dtype: 'i16', shape: [n], data: new Int16Array(n) }
};

describe('recoding positions', () => {
	for (const r of RECODINGS)
		test(`${r.id} keeps the order of the nodes and changes the positions`, () => {
			const out = recodePositions(arrays, n, r.id);
			expect(sameOrder(arrays.pos.data, out.pos.data, n)).toBe(true);
			expect(Array.from(out.pos.data)).not.toEqual(Array.from(arrays.pos.data));
			expect(out.pi).toBe(arrays.pi);
		});
	test('random increasing gives the k-th smallest draw to the k-th smallest position', () => {
		const shuffled: Arrays = { ...arrays, pos: { ...arrays.pos, data: Float32Array.from([3, 1, 2, 0].map((k) => k / 4)) } };
		let s = 0;
		const draws = [0.9, 0.1, 0.5, 0.3];
		const out = recodePositions(shuffled, 4, 'random', () => draws[s++]);
		expect(Array.from(out.pos.data)).toEqual(Array.from(Float32Array.from([0.9, 0.3, 0.5, 0.1])));
	});
});
