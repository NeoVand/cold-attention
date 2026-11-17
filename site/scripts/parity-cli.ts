/**
 * The engine's parity on official instances, from the command line (node; the page's engine code, so the same float
 * arithmetic as in Chrome's V8).
 *   node parity.mjs <static dir> <model name> <task> <split> [limit] [start] [start]
 * Prints one JSON line.
 */
import { parity } from '../src/lib/model/parity';

const [staticDir, name, task, split, limit, start] = process.argv.slice(2);
console.log(JSON.stringify(parity(staticDir, name, task, split, Number(limit ?? 1e9), Number(start ?? 0))));
