/** The palette, Cobalt and Rose, for the canvases and SVG (layout.css holds the same colours as CSS tokens).
 * The paper's rule: a cold block and what it computes (its answer, its bits, its witnesses) in cobalt; the general block
 * (real-valued state carried from loop to loop) in rose; the input and the true answer in ink. */
export const C = {
	paper: '#FFFFFF',
	tint: '#F2F5FB',
	ink: '#1F2A44',
	inkSoft: '#4C5B78',
	blueGrey: '#7D8BA8',
	mist: '#BAC5D9',
	edge: '#DDE3EE',
	rose: '#D6336C',
	roseLight: '#F1A3BC',
	cobalt: '#2557A7',
	cobaltLight: '#9DB8E0'
} as const;

const hex = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const mix = (a: number[], t: number) => a.map((c) => Math.round(255 + (c - 255) * t));
export const rgb = (c: number[]) => `rgb(${c[0]} ${c[1]} ${c[2]})`;
/** the general block: the one-task general block, the shared general block (and its bilinear variant) */
export const isGeneral = (block: string) => block === 'census' || block === 'shared' || block === 'census_bilinear';
/** the colour of a block and of what it computes: rose for the general block, cobalt for a cold block */
export const blockColour = (block: string) => (isGeneral(block) ? { c: C.rose, light: C.roseLight } : { c: C.cobalt, light: C.cobaltLight });
export const rose = (t: number) => rgb(mix(hex(C.rose), Math.max(0, Math.min(1, t))));
export const cobalt = (t: number) => rgb(mix(hex(C.cobalt), Math.max(0, Math.min(1, t))));
