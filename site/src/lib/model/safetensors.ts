/** Read a .safetensors file (float32 tensors only): an 8-byte little-endian header length, a JSON header, then the data. */
export interface Tensor {
	shape: number[];
	data: Float32Array;
}

export function parseSafetensors(buf: ArrayBuffer): Record<string, Tensor> {
	const view = new DataView(buf);
	const len = Number(view.getBigUint64(0, true));
	const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, len))) as Record<
		string,
		{ dtype: string; shape: number[]; data_offsets: [number, number] }
	>;
	const base = 8 + len;
	const out: Record<string, Tensor> = {};
	for (const [name, h] of Object.entries(header)) {
		if (name === '__metadata__') continue;
		if (h.dtype !== 'F32') throw new Error(`${name}: dtype ${h.dtype}, expected F32`);
		const [a, b] = h.data_offsets;
		// copy: the slice may not be 4-byte aligned in the file
		const data = new Float32Array(buf.slice(base + a, base + b));
		out[name] = { shape: h.shape, data };
	}
	return out;
}
