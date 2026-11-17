/** How right one answer is, for one instance, as the library's scorer counts it (masks: F1 over the entries it scores). */
import type { Kind } from '../model/engine';
import type { Arrays } from '../model/inputs';

export interface OutputScore {
	name: string;
	kind: Kind;
	right: number;
	total: number;
	/** the library's per-output score on this instance, in [0, 1] */
	score: number;
}

export function scoreInstance(outputs: { name: string; kind: Kind }[], pred: Record<string, Int32Array>, truth: Arrays, n: number): OutputScore[] {
	return outputs.map(({ name, kind }) => {
		const t = truth[name].data;
		const p = pred[name];
		let right = 0;
		let total = 0;
		if (kind === 'nmask' || kind === 'emask' || kind === 'gmask') {
			let tp = 0, fp = 0, fn = 0;
			for (let i = 0; i < t.length; i++) {
				if (t[i] === -1) continue;
				total++;
				if (p[i] === t[i]) right++;
				if (p[i] === 1 && t[i] === 1) tp++;
				else if (p[i] === 1) fp++;
				else if (t[i] === 1) fn++;
			}
			const pr = tp + fp > 0 ? tp / (tp + fp) : 1;
			const rc = tp + fn > 0 ? tp / (tp + fn) : 1;
			return { name, kind, right, total, score: pr + rc > 0 ? (2 * pr * rc) / (pr + rc) : 0 };
		}
		for (let i = 0; i < t.length; i++) {
			if (kind === 'ecat' && t[i] < 0) continue;
			total++;
			if (p[i] === t[i]) right++;
		}
		void n;
		return { name, kind, right, total, score: total ? right / total : 1 };
	});
}

export const mean = (s: OutputScore[]) => s.reduce((a, b) => a + b.score, 0) / Math.max(1, s.length);
