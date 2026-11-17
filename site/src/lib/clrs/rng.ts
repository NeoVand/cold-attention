/**
 * A small seedable PRNG for the instance samplers: sfc32 (Chris Doty-Humphrey's Small Fast Counting generator,
 * 128-bit state, passes PractRand) seeded through splitmix32. Doubles carry 53 random bits, as numpy's do.
 *
 * Same seed, same stream, on every platform. The stream is NOT numpy's MT19937: a sampled instance has the
 * library's distribution, not the library's exact numbers.
 */

export interface Rng {
	/** A uniform 32-bit unsigned integer. */
	u32(): number;
	/** A uniform double in [0, 1) with 53 random bits. */
	random(): number;
	/** A uniform double in [low, high), computed as numpy does: low + (high - low) * random(). */
	uniform(low: number, high: number): number;
	/** A uniform integer in [0, n), unbiased. */
	int(n: number): number;
	/** 1 with probability p, else 0 (numpy's binomial(1, p)). */
	bernoulli(p: number): number;
	/** A uniformly random permutation of 0..n-1 (Fisher-Yates). */
	permutation(n: number): Int32Array;
	/** One element of a non-empty list, uniformly. */
	choice<T>(items: readonly T[]): T;
}

function splitmix32(seed: number): () => number {
	let s = seed >>> 0;
	return () => {
		s = (s + 0x9e3779b9) >>> 0;
		let z = s;
		z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
		z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
		return (z ^ (z >>> 16)) >>> 0;
	};
}

/** A generator seeded by any number (its low 32 bits and, for non-integers, the rest of its bits). */
export function makeRng(seed: number): Rng {
	const view = new DataView(new ArrayBuffer(8));
	view.setFloat64(0, seed);
	const sm = splitmix32((view.getUint32(0) ^ Math.imul(view.getUint32(4), 0x27d4eb2d)) >>> 0);
	let a = sm();
	let b = sm();
	let c = sm();
	let d = sm();
	const u32 = (): number => {
		const t = (((a + b) >>> 0) + d) >>> 0;
		d = (d + 1) >>> 0;
		a = b ^ (b >>> 9);
		b = (c + (c << 3)) >>> 0;
		c = ((c << 21) | (c >>> 11)) >>> 0;
		c = (c + t) >>> 0;
		return t;
	};
	for (let i = 0; i < 12; i++) u32();
	const random = (): number => ((u32() >>> 5) * 67108864 + (u32() >>> 6)) / 9007199254740992;
	const int = (n: number): number => {
		if (!(n >= 1) || !Number.isInteger(n) || n > 0x100000000) throw new RangeError(`int(${n})`);
		const limit = 0x100000000 - (0x100000000 % n);
		for (;;) {
			const x = u32();
			if (x < limit) return x % n;
		}
	};
	return {
		u32,
		random,
		uniform: (low, high) => low + (high - low) * random(),
		int,
		bernoulli: (p) => (random() < p ? 1 : 0),
		permutation: (n) => {
			const p = new Int32Array(n);
			for (let i = 0; i < n; i++) p[i] = i;
			for (let i = n - 1; i > 0; i--) {
				const j = int(i + 1);
				const t = p[i];
				p[i] = p[j];
				p[j] = t;
			}
			return p;
		},
		choice: (items) => {
			if (items.length === 0) throw new RangeError('choice of an empty list');
			return items[int(items.length)];
		}
	};
}
