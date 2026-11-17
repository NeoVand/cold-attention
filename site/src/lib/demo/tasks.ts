/** The 30 tasks as the page shows them: a family (which view draws it) and one plain sentence each. */
export type Family = 'graph' | 'array' | 'table' | 'interval' | 'string' | 'geometry';

export interface TaskInfo {
	id: string;
	name: string;
	family: Family;
	group: string;
	/** what the block is asked, in plain words */
	asks: string;
	/** item count of a fresh instance: default and range (the official test is 64; training saw at most 16) */
	size: [number, number, number];
}

const T = (id: string, name: string, family: Family, group: string, asks: string, size: [number, number, number] = [16, 4, 64]): TaskInfo => ({
	id,
	name,
	family,
	group,
	asks,
	size
});

export const TASKS: TaskInfo[] = [
	T('bfs', 'BFS', 'graph', 'Graphs', 'Each node points to its parent in the breadth-first tree from the source.'),
	T('dfs', 'DFS', 'graph', 'Graphs', 'Each node points to its parent in the depth-first forest (smallest index first).'),
	T('topological_sort', 'Topological Sort', 'graph', 'Graphs', 'Each node points to the next node in a topological order of the DAG; the first node is marked.'),
	T('articulation_points', 'Articulation Points', 'graph', 'Graphs', 'Which nodes disconnect the graph when removed.'),
	T('bridges', 'Bridges', 'graph', 'Graphs', 'Which edges disconnect the graph when removed.'),
	T('strongly_connected_components', 'SCC', 'graph', 'Graphs', 'Each node points to a representative of its strongly connected component.'),
	T('mst_kruskal', 'MST-Kruskal', 'graph', 'Graphs', 'Which edges belong to the minimum spanning tree.'),
	T('mst_prim', 'MST-Prim', 'graph', 'Graphs', 'Each node points to its parent in the minimum spanning tree grown from the source.'),
	T('bellman_ford', 'Bellman-Ford', 'graph', 'Graphs', 'Each node points to its predecessor on a shortest path from the source.'),
	T('dijkstra', 'Dijkstra', 'graph', 'Graphs', 'Each node points to its predecessor on a shortest path from the source.'),
	T('dag_shortest_paths', 'DAG Shortest Paths', 'graph', 'Graphs', 'Each node points to its predecessor on a shortest path from the source, in a DAG.'),
	T('floyd_warshall', 'Floyd-Warshall', 'table', 'Graphs', 'For every pair, the node just before the target on a shortest path.'),
	T('insertion_sort', 'Insertion Sort', 'array', 'Sorting', 'Each number points to the one just below it in sorted order (the smallest to itself).'),
	T('bubble_sort', 'Bubble Sort', 'array', 'Sorting', 'Each number points to the one just below it in sorted order (the smallest to itself).'),
	T('heapsort', 'Heapsort', 'array', 'Sorting', 'Each number points to the one just below it in sorted order (the smallest to itself).'),
	T('quicksort', 'Quicksort', 'array', 'Sorting', 'Each number points to the one just below it in sorted order (the smallest to itself).'),
	T('minimum', 'Minimum', 'array', 'Searching', 'Which number is the smallest.'),
	T('quickselect', 'Quickselect', 'array', 'Searching', 'Which number is the median.'),
	T('binary_search', 'Binary Search', 'array', 'Searching', 'The first position whose number is at least the target.'),
	T('find_maximum_subarray_kadane', 'Find Max. Subarray', 'array', 'Dynamic programming', 'Where the run of numbers with the largest sum starts and ends.'),
	T('matrix_chain_order', 'Matrix Chain Order', 'table', 'Dynamic programming', 'For every span of matrices, where to split the product to multiply cheapest.'),
	T('optimal_bst', 'Optimal BST', 'table', 'Dynamic programming', 'For every span of keys, which key is the root of the cheapest search tree.'),
	T('lcs_length', 'LCS Length', 'table', 'Dynamic programming', 'For every pair of positions, one in each string, the arrow of the longest-common-subsequence table.'),
	T('activity_selector', 'Activity Selector', 'interval', 'Greedy', 'The largest set of activities that do not overlap.'),
	T('task_scheduling', 'Task Scheduling', 'interval', 'Greedy', 'The heaviest set of unit tasks that meet their deadlines.'),
	T('naive_string_matcher', 'Naive String Match', 'string', 'Strings', 'Where the needle first occurs in the haystack.', [20, 8, 80]),
	T('kmp_matcher', 'KMP Matcher', 'string', 'Strings', 'Where the needle first occurs in the haystack.', [20, 8, 80]),
	T('segments_intersect', 'Segments Intersect', 'geometry', 'Geometry', 'Whether two segments cross.', [4, 4, 4]),
	T('graham_scan', 'Graham Scan', 'geometry', 'Geometry', 'Which points lie on the convex hull.'),
	T('jarvis_march', "Jarvis' March", 'geometry', 'Geometry', 'Which points lie on the convex hull.')
];

export const TASK = Object.fromEntries(TASKS.map((t) => [t.id, t])) as Record<string, TaskInfo>;
export const GROUPS = [...new Set(TASKS.map((t) => t.group))];

export const BLOCK_LABEL: Record<string, string> = {
	census: 'General block',
	cold: 'Cold block',
	compare_first: 'Cold block, numbers compared',
	pair_numbers: 'Cold block, pair numbers',
	shared: 'Shared general block'
};
export const BLOCK_NOTE: Record<string, string> = {
	census: 'the input\'s numbers enter as embeddings and the state is carried from loop to loop; read after 8 loops',
	cold: 'only bits enter the state, which is rebuilt from the input and the bits every loop; read when they stop changing, or after n loops for n items',
	compare_first: 'numbers stay in registers and enter only as comparisons; read when its bits stop changing',
	pair_numbers: 'also builds one exact number per pair; read when bits and numbers stop changing',
	shared: 'one general block shared by all 30 tasks, each with its own input encoder and read-outs; read after 8 loops (depth-first search after n)'
};

/**
 * The block each task opens on: the one its symmetry calls for. Compare-first where the answer depends on the
 * numbers only through their order; exact pair numbers where it depends on path sums or bottlenecks; the census block
 * elsewhere (and as a toggle wherever another block exists). Depth-first search opens on the compare-first block with
 * the positions in its register; topological sort and strongly connected components on the cold block read after n
 * loops (on strongly connected components it scores below the census block; the page shows both).
 */
export const PREFERRED_BLOCK: Record<string, string> = {
	insertion_sort: 'compare_first',
	bubble_sort: 'compare_first',
	heapsort: 'compare_first',
	quicksort: 'compare_first',
	minimum: 'compare_first',
	binary_search: 'compare_first',
	activity_selector: 'compare_first',
	quickselect: 'compare_first',
	task_scheduling: 'compare_first',
	bellman_ford: 'pair_numbers',
	dijkstra: 'pair_numbers',
	dag_shortest_paths: 'pair_numbers',
	mst_kruskal: 'pair_numbers',
	floyd_warshall: 'pair_numbers',
	mst_prim: 'pair_numbers',
	dfs: 'compare_first',
	topological_sort: 'cold',
	strongly_connected_components: 'cold'
};

/** FloydNet (arXiv 2601.19094), Table 9: one model per task, best of runs with and without hints (average 96.64). */
export const FLOYDNET: Record<string, number> = {
	activity_selector: 95.3, articulation_points: 100, bellman_ford: 100, bfs: 100, binary_search: 94.47, bridges: 100,
	bubble_sort: 100, dag_shortest_paths: 100, dfs: 100, dijkstra: 97.67, find_maximum_subarray_kadane: 86.2,
	floyd_warshall: 96.2, graham_scan: 99.63, heapsort: 100, insertion_sort: 100, jarvis_march: 99.5, kmp_matcher: 99.43,
	lcs_length: 95.7, matrix_chain_order: 91.5, minimum: 100, mst_kruskal: 99.63, mst_prim: 99.67, naive_string_matcher: 100,
	optimal_bst: 82.87, quickselect: 80.33, quicksort: 100, segments_intersect: 99.9, strongly_connected_components: 90.07,
	task_scheduling: 91.05, topological_sort: 100
};

/**
 * The recoding test on the page: per task, a block that reads positions only through comparisons and one that embeds
 * them. live: shown to every reader (otherwise only with ?recode in the address, until the paper's own reads confirm
 * that recodings move none of the compare-first block's answers).
 */
export const RECODE: Record<string, { live: boolean; note?: string; models: { name: string; label: string; note: string }[] }> = {
	dfs: {
		live: true,
		note:
			'Measured on all 32 official test graphs (2,048 parents) and three seeds: with positions compared the cold block moves no ' +
			'parent under pos / 2, √pos or pos²; with positions encoded it moves up to 669 under pos². On fresh graphs of 96 and 128 ' +
			'nodes the first scores 100 on every seed, the second 58 to 84.',
		models: [
			{ name: 'compare_first/dfs/seed0', label: 'Cold block, positions compared', note: 'positions in its register, read only through comparisons' },
			{ name: 'cold/dfs/seed0', label: 'Cold block, positions encoded', note: 'positions in its input embedding' }
		]
	}
};

/** Models shown in the grid and on the page only with a caveat: still being worked on (fail at 64 items). */
export const PROVISIONAL: Record<string, string[]> = {};
