/**
 * Arrows between circles, drawn as a shaft and a head shape (not SVG markers, whose size follows the stroke and whose ids
 * clash between views). The arrow is a quadratic curve from a's centre to b's, bent to the left by `bend` times the chord,
 * clipped `gap` outside each circle; the head's tip sits on b's clipped end, along the curve's last direction. When the
 * circles leave less room than the head needs, the head shrinks to fit; when they nearly touch, there is no arrow.
 */
type P = [number, number];

const at = (p0: P, q: P, p2: P, t: number): P => {
	const u = 1 - t;
	return [u * u * p0[0] + 2 * u * t * q[0] + t * t * p2[0], u * u * p0[1] + 2 * u * t * q[1] + t * t * p2[1]];
};
const dist = (a: P, b: P) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/** the parameter in [lo, hi] where the distance from `from` crosses r (distance grows from lo to hi) */
function cross(p0: P, q: P, p2: P, from: P, r: number, lo: number, hi: number): number {
	for (let k = 0; k < 30; k++) {
		const m = (lo + hi) / 2;
		if (dist(at(p0, q, p2, m), from) < r) lo = m;
		else hi = m;
	}
	return (lo + hi) / 2;
}

export interface Arrow {
	/** the shaft (empty when the head fills the room) */
	shaft: string;
	/** the head, a closed shape to fill */
	head: string;
}

/** The control point of the bent curve from a to b. */
export function control(a: P, b: P, bend: number): P {
	return [(a[0] + b[0]) / 2 - (b[1] - a[1]) * bend, (a[1] + b[1]) / 2 + (b[0] - a[0]) * bend];
}

/** A curve from circle a (radius ra) to circle b (radius rb), clipped `gap` outside both, with no head. */
export function link(a: P, b: P, ra: number, rb: number, bend = 0, gap = 1.5): string | null {
	const q = control(a, b, bend);
	if (dist(a, b) <= ra + rb + 2 * gap) return null;
	const t0 = cross(a, q, b, a, ra + gap, 0, 0.5);
	const t1 = 1 - cross(b, q, a, b, rb + gap, 0, 0.5);
	return seg(a, q, b, t0, t1);
}

/** An arrow from circle a to circle b; `head` is the head's length (its width is 0.9 of that). */
export function arrow(a: P, b: P, ra: number, rb: number, bend: number, head: number, gap = 1.5): Arrow | null {
	const q = control(a, b, bend);
	const room = dist(a, b) - ra - rb - 2 * gap;
	if (room < 3) return null;
	const t0 = cross(a, q, b, a, ra + gap, 0, 0.5);
	const tTip = 1 - cross(b, q, a, b, rb + gap, 0, 0.5);
	const tip = at(a, q, b, tTip);
	const hl = Math.min(head, 0.75 * dist(tip, at(a, q, b, t0)));
	// the head's back: where the curve is hl from the tip (searching back from the tip)
	let lo = t0;
	let hi = tTip;
	for (let k = 0; k < 30; k++) {
		const m = (lo + hi) / 2;
		if (dist(at(a, q, b, m), tip) > hl) lo = m;
		else hi = m;
	}
	const back = at(a, q, b, (lo + hi) / 2);
	const ux = (tip[0] - back[0]) / (dist(tip, back) || 1);
	const uy = (tip[1] - back[1]) / (dist(tip, back) || 1);
	const hw = 0.45 * hl;
	const notch: P = [tip[0] - ux * hl * 0.78, tip[1] - uy * hl * 0.78];
	const f = (p: P) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`;
	const head_ = `M${f(tip)} L${f([tip[0] - ux * hl - uy * hw, tip[1] - uy * hl + ux * hw])} L${f(notch)} L${f([tip[0] - ux * hl + uy * hw, tip[1] - uy * hl - ux * hw])} Z`;
	// the shaft runs into the head's notch, so no gap shows between them
	let tn = (lo + hi) / 2;
	for (let k = 0, l2 = t0, h2 = tTip; k < 30; k++) {
		const m = (l2 + h2) / 2;
		if (dist(at(a, q, b, m), tip) > hl * 0.7) l2 = m;
		else h2 = m;
		tn = (l2 + h2) / 2;
	}
	return { shaft: tn > t0 ? seg(a, q, b, t0, tn) : '', head: head_ };
}

/** The piece of the quadratic (p0, q, p2) between parameters s and e, as a path. */
function seg(p0: P, q: P, p2: P, s: number, e: number): string {
	const a = at(p0, q, p2, s);
	const b = at(p0, q, p2, e);
	// the blossom of the two parameters is the piece's control point
	const w0 = (1 - s) * (1 - e);
	const w1 = (1 - s) * e + s * (1 - e);
	const w2 = s * e;
	const c: P = [w0 * p0[0] + w1 * q[0] + w2 * p2[0], w0 * p0[1] + w1 * q[1] + w2 * p2[1]];
	const f = (p: P) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`;
	return `M${f(a)} Q${f(c)} ${f(b)}`;
}

/** The head length for nodes of radius r. */
export const headFor = (r: number) => Math.max(6, Math.min(12, 2 + 0.9 * r));
