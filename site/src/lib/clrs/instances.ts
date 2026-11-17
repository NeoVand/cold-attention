/**
 * CLRS-30 instances: the 30 tasks, the layout of the exported split files, a decoder, one-instance access and the
 * inverse (packing instances back into the padded layout the model reads).
 *
 * A split file (static/instances/<task>/<split>.json) is {task, split, count, arrays: {name: {dtype, shape, b64}}}, each
 * array the padded numpy array, base64 little-endian. S instances, padded size M, n[s] real items; padding beyond n
 * (0 for inputs, -1 for outputs).
 *
 * An unpadded Instance keeps every array's dtype and names inputs and outputs by the library's names (specs.SPECS):
 *   node   one value per item, length n
 *   edge   one value per ordered pair, n*n row-major ([i * n + j] is the pair i -> j)
 *   cat    a one-hot category per item, n*C row-major (C = 4 for the string tasks' 'key')
 *   graph  one number per instance (a source index, a target, a mask_one's index, a graph mask)
 */

export const GRAPH_TASKS = [
	'bfs',
	'bellman_ford',
	'dijkstra',
	'dag_shortest_paths',
	'floyd_warshall',
	'mst_prim',
	'topological_sort',
	'dfs',
	'strongly_connected_components',
	'articulation_points',
	'bridges',
	'mst_kruskal'
] as const;

export const SEQ_TASKS = [
	'insertion_sort',
	'bubble_sort',
	'heapsort',
	'quicksort',
	'minimum',
	'quickselect',
	'binary_search',
	'find_maximum_subarray_kadane',
	'matrix_chain_order',
	'optimal_bst',
	'lcs_length',
	'activity_selector',
	'task_scheduling',
	'naive_string_matcher',
	'kmp_matcher',
	'segments_intersect',
	'graham_scan',
	'jarvis_march'
] as const;

export const TASKS = [...GRAPH_TASKS, ...SEQ_TASKS] as const;
export type GraphTask = (typeof GRAPH_TASKS)[number];
export type Task = (typeof TASKS)[number];

export function isTask(name: string): name is Task {
	return (TASKS as readonly string[]).includes(name);
}

export function isGraph(task: Task): task is GraphTask {
	return (GRAPH_TASKS as readonly string[]).includes(task);
}

export type DType = 'f32' | 'f64' | 'i16' | 'u8';
export type TypedArray = Float32Array | Float64Array | Int16Array | Uint8Array;
export type Kind = 'node' | 'edge' | 'cat' | 'graph';
/** An instance value: a number for kind 'graph', a typed array otherwise. */
export type Value = number | TypedArray;

export interface Field {
	/** The library's name (specs.SPECS). */
	name: string;
	/** The array's name in a split file (non-graph tasks prefix inputs with 'x_'). */
	file: string;
	kind: Kind;
	dtype: DType;
	/** Categories per item (kind 'cat' only). */
	classes?: number;
}

export interface Schema {
	inputs: Field[];
	outputs: Field[];
}

const SOURCE_TASKS: readonly Task[] = ['bfs', 'bellman_ford', 'dijkstra', 'dag_shortest_paths', 'mst_prim'];

function graphSchema(task: GraphTask): Schema {
	const inputs: Field[] = [
		{ name: 'pos', file: 'pos', kind: 'node', dtype: 'f32' },
		{ name: 'A', file: 'A', kind: 'edge', dtype: 'f32' },
		{ name: 'adj', file: 'adj', kind: 'edge', dtype: 'u8' }
	];
	if (SOURCE_TASKS.includes(task)) inputs.push({ name: 's', file: 's', kind: 'graph', dtype: 'i16' });
	const out = (name: string, kind: Kind): Field => ({ name, file: name, kind, dtype: 'i16' });
	const outputs: Record<GraphTask, Field[]> = {
		bfs: [out('pi', 'node')],
		bellman_ford: [out('pi', 'node')],
		dijkstra: [out('pi', 'node')],
		dag_shortest_paths: [out('pi', 'node')],
		floyd_warshall: [out('Pi', 'edge')],
		mst_prim: [out('pi', 'node')],
		topological_sort: [out('topo', 'node'), out('topo_head', 'graph')],
		dfs: [out('pi', 'node')],
		strongly_connected_components: [out('scc_id', 'node')],
		articulation_points: [out('is_cut', 'node')],
		bridges: [out('is_bridge', 'edge')],
		mst_kruskal: [out('in_mst', 'edge')]
	};
	return { inputs, outputs: outputs[task] };
}

const num = (name: string): Field => ({ name, file: 'x_' + name, kind: 'node', dtype: 'f64' });
const target: Field = { name: 'target', file: 'x_target', kind: 'graph', dtype: 'f64' };
const strings: Field[] = [
	{ name: 'string', file: 'x_string', kind: 'node', dtype: 'u8' },
	num('pos'),
	{ name: 'key', file: 'x_key', kind: 'cat', dtype: 'u8', classes: 4 }
];
const o = (name: string, kind: Kind): Field => ({ name, file: name, kind, dtype: 'i16' });

const SEQ_SCHEMA: Record<Exclude<Task, GraphTask>, Schema> = {
	insertion_sort: { inputs: [num('pos'), num('key')], outputs: [o('pred', 'node')] },
	bubble_sort: { inputs: [num('pos'), num('key')], outputs: [o('pred', 'node')] },
	heapsort: { inputs: [num('pos'), num('key')], outputs: [o('pred', 'node')] },
	quicksort: { inputs: [num('pos'), num('key')], outputs: [o('pred', 'node')] },
	minimum: { inputs: [num('pos'), num('key')], outputs: [o('min', 'graph')] },
	quickselect: { inputs: [num('pos'), num('key')], outputs: [o('median', 'graph')] },
	binary_search: { inputs: [num('pos'), num('key'), target], outputs: [o('return', 'graph')] },
	find_maximum_subarray_kadane: {
		inputs: [num('pos'), num('key')],
		outputs: [o('start', 'graph'), o('end', 'graph')]
	},
	matrix_chain_order: { inputs: [num('pos'), num('p')], outputs: [o('s', 'edge')] },
	optimal_bst: { inputs: [num('pos'), num('p'), num('q')], outputs: [o('root', 'edge')] },
	lcs_length: { inputs: strings, outputs: [o('b', 'edge')] },
	activity_selector: { inputs: [num('pos'), num('s'), num('f')], outputs: [o('selected', 'node')] },
	task_scheduling: { inputs: [num('pos'), num('d'), num('w')], outputs: [o('selected', 'node')] },
	naive_string_matcher: { inputs: strings, outputs: [o('match', 'graph')] },
	kmp_matcher: { inputs: strings, outputs: [o('match', 'graph')] },
	segments_intersect: { inputs: [num('pos'), num('x'), num('y')], outputs: [o('intersect', 'graph')] },
	graham_scan: { inputs: [num('pos'), num('x'), num('y')], outputs: [o('in_hull', 'node')] },
	jarvis_march: { inputs: [num('pos'), num('x'), num('y')], outputs: [o('in_hull', 'node')] }
};

/** Inputs and outputs of a task, in the order the split files list them (after 'n'). */
export function schema(task: Task): Schema {
	return isGraph(task) ? graphSchema(task) : SEQ_SCHEMA[task];
}

// ---------------------------------------------------------------------------------------------------------------
// Split files

export interface ArrayJson {
	dtype: DType;
	shape: number[];
	b64: string;
}

export interface SplitJson {
	task: string;
	split: string;
	count: number;
	arrays: Record<string, ArrayJson>;
}

export interface NdArray {
	dtype: DType;
	shape: number[];
	data: TypedArray;
}

export interface Split {
	task: Task;
	split: string;
	count: number;
	/** Padded size: items per instance in the arrays. */
	M: number;
	/** Every array of the file by its file name ('n', 'pos', 'x_key', 'pi', ...). */
	arrays: Record<string, NdArray>;
}

export interface Instance {
	task: Task;
	/** Items: graph nodes, array elements, both strings' characters, optimal_bst's n + 1 keys, 4 endpoints. */
	n: number;
	inputs: Record<string, Value>;
	outputs: Record<string, Value>;
}

const BYTES: Record<DType, number> = { f32: 4, f64: 8, i16: 2, u8: 1 };
const LITTLE = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

export function allocate(dtype: DType, length: number): TypedArray {
	switch (dtype) {
		case 'f32':
			return new Float32Array(length);
		case 'f64':
			return new Float64Array(length);
		case 'i16':
			return new Int16Array(length);
		case 'u8':
			return new Uint8Array(length);
	}
}

function base64Bytes(b64: string): Uint8Array {
	const bin = atob(b64);
	const bytes = new Uint8Array(bin.length);
	for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
	return bytes;
}

function bytesBase64(bytes: Uint8Array): string {
	let s = '';
	const CH = 0x8000;
	for (let i = 0; i < bytes.length; i += CH) s += String.fromCharCode(...bytes.subarray(i, i + CH));
	return btoa(s);
}

/** Decodes one array of a split file (base64 little-endian) into a typed array. */
export function decodeArray(a: ArrayJson): NdArray {
	const bytes = base64Bytes(a.b64);
	const size = a.shape.reduce((p, q) => p * q, 1);
	if (bytes.length !== size * BYTES[a.dtype]) throw new Error(`array of ${bytes.length} bytes, shape ${a.shape}`);
	let data: TypedArray;
	if (LITTLE || a.dtype === 'u8') {
		const buf = bytes.buffer;
		data =
			a.dtype === 'f32'
				? new Float32Array(buf)
				: a.dtype === 'f64'
					? new Float64Array(buf)
					: a.dtype === 'i16'
						? new Int16Array(buf)
						: bytes;
	} else {
		const view = new DataView(bytes.buffer);
		data = allocate(a.dtype, size);
		for (let i = 0; i < size; i++) {
			if (a.dtype === 'f32') data[i] = view.getFloat32(4 * i, true);
			else if (a.dtype === 'f64') data[i] = view.getFloat64(8 * i, true);
			else data[i] = view.getInt16(2 * i, true);
		}
	}
	return { dtype: a.dtype, shape: [...a.shape], data };
}

/** Encodes an array as a split file stores it. */
export function encodeArray(a: NdArray): ArrayJson {
	const size = a.data.length;
	const bytes = new Uint8Array(size * BYTES[a.dtype]);
	const view = new DataView(bytes.buffer);
	for (let i = 0; i < size; i++) {
		const v = a.data[i];
		if (a.dtype === 'f32') view.setFloat32(4 * i, v, true);
		else if (a.dtype === 'f64') view.setFloat64(8 * i, v, true);
		else if (a.dtype === 'i16') view.setInt16(2 * i, v, true);
		else view.setUint8(i, v);
	}
	return { dtype: a.dtype, shape: [...a.shape], b64: bytesBase64(bytes) };
}

/** Decodes a split file's JSON. */
export function decodeSplit(json: SplitJson): Split {
	if (!isTask(json.task)) throw new Error(`unknown task ${json.task}`);
	const arrays: Record<string, NdArray> = {};
	for (const [name, a] of Object.entries(json.arrays)) arrays[name] = decodeArray(a);
	const sch = schema(json.task);
	const node = [...sch.inputs, ...sch.outputs].find((f) => f.kind !== 'graph');
	if (!node) throw new Error(`no per-item array in ${json.task}`);
	const M = arrays[node.file].shape[1];
	return { task: json.task, split: json.split, count: json.count, M, arrays };
}

function slice(a: NdArray, f: Field, i: number, n: number): Value {
	const s = a.shape;
	if (f.kind === 'graph') return a.data[i];
	if (f.kind === 'node') {
		const M = s[1];
		return a.data.slice(i * M, i * M + n);
	}
	const out = allocate(f.dtype, f.kind === 'edge' ? n * n : n * s[2]);
	if (f.kind === 'edge') {
		const M = s[1];
		for (let r = 0; r < n; r++) out.set(a.data.subarray(i * M * M + r * M, i * M * M + r * M + n), r * n);
	} else {
		const M = s[1];
		const C = s[2];
		out.set(a.data.subarray(i * M * C, i * M * C + n * C));
	}
	return out;
}

/** Instance i of a split, unpadded (copies). */
export function instanceAt(split: Split, i: number): Instance {
	if (!(i >= 0 && i < split.count)) throw new RangeError(`instance ${i} of ${split.count}`);
	const n = split.arrays.n.data[i];
	const sch = schema(split.task);
	const inputs: Record<string, Value> = {};
	const outputs: Record<string, Value> = {};
	for (const f of sch.inputs) inputs[f.name] = slice(split.arrays[f.file], f, i, n);
	for (const f of sch.outputs) outputs[f.name] = slice(split.arrays[f.file], f, i, n);
	return { task: split.task, n, inputs, outputs };
}

/** Every instance of a split. */
export function instances(split: Split): Instance[] {
	return Array.from({ length: split.count }, (_, i) => instanceAt(split, i));
}

/**
 * Packs instances of one task into the padded layout of a split file (witness_clrs.generate's pack / pack_items): arrays by file name,
 * 'n' first, then inputs and outputs in file order; inputs padded with 0, per-item and per-pair outputs with -1.
 * Outputs missing from an instance are left out of the result. M defaults to the largest n.
 */
export function toPadded(list: readonly Instance[], M?: number): Record<string, NdArray> {
	if (list.length === 0) throw new Error('no instances');
	const task = list[0].task;
	const size = M ?? Math.max(...list.map((x) => x.n));
	const S = list.length;
	const out: Record<string, NdArray> = {};
	const n = new Int16Array(S);
	list.forEach((x, s) => {
		if (x.task !== task) throw new Error(`mixed tasks ${task} and ${x.task}`);
		if (x.n > size) throw new Error(`instance of ${x.n} items does not fit M = ${size}`);
		n[s] = x.n;
	});
	out.n = { dtype: 'i16', shape: [S], data: n };
	const sch = schema(task);
	const pack = (f: Field, values: (Value | undefined)[], pad: number) => {
		if (values.some((v) => v === undefined)) return;
		const C = f.classes ?? 0;
		const shape =
			f.kind === 'graph' ? [S] : f.kind === 'node' ? [S, size] : f.kind === 'edge' ? [S, size, size] : [S, size, C];
		const data = allocate(f.dtype, shape.reduce((p, q) => p * q, 1));
		if (f.kind !== 'graph') data.fill(pad);
		values.forEach((v, s) => {
			const k = list[s].n;
			if (f.kind === 'graph') {
				data[s] = v as number;
				return;
			}
			const a = v as TypedArray;
			if (f.kind === 'node') data.set(a.subarray(0, k), s * size);
			else if (f.kind === 'cat') data.set(a.subarray(0, k * C), s * size * C);
			else for (let r = 0; r < k; r++) data.set(a.subarray(r * k, r * k + k), s * size * size + r * size);
		});
		out[f.file] = { dtype: f.dtype, shape, data };
	};
	for (const f of sch.inputs)
		pack(
			f,
			list.map((x) => x.inputs[f.name]),
			0
		);
	for (const f of sch.outputs)
		pack(
			f,
			list.map((x) => x.outputs[f.name]),
			-1
		);
	return out;
}
