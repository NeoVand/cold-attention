# The paper's numbers

```bash
pip install huggingface_hub
python numbers/numbers.py                  # reads the small JSON files of the model repositories on Hugging Face (about 12 MB)
python numbers/numbers.py --local DIR      # or a local copy of the repositories
```

The paper prints 1,433 numbers. The script prints every one beside the value the paper prints, with its source:

- **Recomputed, 845 numbers, all equal to the paper's.**
  - 683 come from the released models. They are computed from the scores each released model records in its `config.json`. Every released model was read again with this repository's code and gives exactly the predictions it made when it was trained ([REPRODUCE.md](../REPRODUCE.md)), so these scores are the models' own.
  - The script recomputes the averages, seed ranges, task counts, wins, ties and losses, and the differences between settings.
  - The families: the one-task general block, the cold blocks, the setting the theory prescribes, the shared blocks, the comparisons with models trained without hints, the block variants, item memory, Max Retrieval, the constraint problems, the block beside a frozen language model (comparison and kinship), the depth study, the committee-chosen training of strong components, and CLUTRR.
  - 162 come from other papers' numbers: FloydNet's Tables 9, 10 and 12, the best other model per task, the CLRS-30 models trained without hints, the published Max Retrieval and CLUTRR results. They are in [data/published.json](data/published.json) with their sources.
- **Read, 588 numbers, from the files of measured values in [data/measured/](data/measured).** These are measurements that are not stored model scores. Each file says what was measured and holds its numbers as the paper prints them, so they are shown, not checked.

| File | Numbers | What was measured |
|---|---:|---|
| `ablation_ladder` | 37 | the block with every channel warm, with the witness sum soft, and cold (seed 0) |
| `activations` | 7 | probes of the general block's activations |
| `answer_depth` | 19 | answer depths, in hops, in the test files and the deeper and chain graphs |
| `block` | 9 | milliseconds per loop, and a worked example of the block's arithmetic |
| `certificates` | 39 | certificates of exactness at every size for the organ networks, and the Lean proofs |
| `clutrr_analysis` | 54 | earlier builds of the graph trainer, the read-in's reading of the stories, rule tables learned from text, the distilled and untrained read-ins |
| `cold_blocks_extra` | 39 | the cold blocks on the second, fresh test file; training lengths and times; the built register switched off; fresh 128-node graphs |
| `constraint_problems_extra` | 6 | 2-SAT contradiction depths at 500 variables, and the first loop at which every instance is right |
| `costs` | 11 | what the runs cost |
| `depth_first_search` | 70 | the general block read after n loops, the cold blocks' training checks, margins by depth, graphs of 96 to 192 nodes, recodings |
| `forecasts` | 6 | forecasts registered before the runs, and how they scored |
| `frontier_models` | 20 | a frontier language model on CLRS-30 instances at 16 and 64 items |
| `general_block_runs` | 5 | the one-task general block's fresh test file, training updates and minutes, one parameter count, the best block per task |
| `item_memory_extra` | 16 | item memory's bilinear read-out arm, its number of runs and cost |
| `kinship_reading` | 6 | the frozen models' reading of single sentences, and the wordings |
| `listing` | 8 | listing number programs for optimal BST |
| `maps` | 44 | shortest routes on maps of 49 and 250 towns, loop by loop |
| `max_retrieval_analysis` | 7 | class offsets in the softmax models' key score, the offset-only model of the cold read |
| `quickselect` | 13 | runs asked for the answer by loop two, validated one size up, and read between the training sizes |
| `recodings` | 23 | reads with the inputs' numbers recoded by an increasing function |
| `segments` | 5 | segment intersection trained four times longer; the product pair number with the form given |
| `set` | 26 | SET at a fixed temperature, and its joins read warm and cold |
| `shared_blocks_extra` | 59 | one seed of a shared block at 80,000 updates; the general-and-cold block with a half or the coupling switched off; fresh files; cost and hours; forecasts |
| `strings` | 24 | the string matchers with positions compared, texts over 2 to 4 letters, the stored-count program without open tests |
| `strong_components` | 20 | fresh-file reads, graphs with five or more entry points, 96 and 128 nodes, the committee's cost |
| `task_scheduling` | 3 | the general block with deadlines in units of n and positions as given in training |
| `worked_examples` | 12 | the first figure's worked examples (the small map and the SET deal) |

The paper's values are in [data/paper.json](data/paper.json), one entry per printed number, under the keys used here.

One count needs a note. The paper gives each one-task model 185,720 to 194,876 parameters, "counting an edge read-out that only edge tasks use". The released models keep only what their outputs read: 152,440 to 193,112 parameters. The script adds back what the training code carries and recomputes the paper's count:

- the edge read-out;
- the node-pointer and mask vectors no output reads;
- two 64-vectors the block never reads.
