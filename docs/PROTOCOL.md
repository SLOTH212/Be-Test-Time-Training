# Scientific protocol

## NTP target and update
Let h[t,i] be the MLP intermediate activation and c[t+1,d] the contextual post-attention normalized hidden state of the next token in the same chunk. The learned per-channel gate g is initialized to zero. With learned projection P, the target is y[t,e] = sum_d c[t+1,d] g[d] P[d,e]. The down-projection update is delta[e,i] = learning_rate * sum_t y[t,e] h[t,i], over valid adjacent pairs. The source implementation defines dtype conversion and clipping; the release preserves it rather than replacing it with a scalar-loss gradient.

The current chunk uses W plus its existing accumulated state. Only afterward is its new delta constructed and accumulated. Inference learning rate is 1 and the Frobenius delta clip is 1e-5. A new sample clears fast weights and KV state. Generation performs no updates. A partial inference prompt tail is forward-only. Training permits valid adjacent pairs in a partial document tail; this is a deliberate distinction.

## Action space
OFF updates no configured layer; Lk updates only layer k; ALL updates every configured TTT layer. Configured layers are ordered, unique, nonnegative integers. Qwen3-1.7B uses [0,6,12,18,24], 16,384 context, 1,024 chunks. The 32,768/4,096 scale-up uses [0,6,12,18,24,30]. Qwen's historical and scale-up cores retain their separate chunk guards.

## Baselines and search
Best Single maximizes the global mean among single-layer actions only, with configured-order tie handling. Sample Best is the sample-wise maximum over the entire action set, including OFF and ALL. Dynamic LB begins with constant sequences, uses beam 4 and one forward-position sweep, and replays each changed sequence from the original state. It is a feasible recomputed lower bound on the unrestricted sequence optimum; it is not exhaustive search.

## Cohorts
Fixed: 13 tasks, 6,500 samples. Dynamic: 12 tasks, 6,000 samples after the frozen CWE exclusion. Sample Best already reaches the ceiling on 4,565 samples; 1,435 are searched. Among searched samples, 173 improve, 1,262 tie, and zero worsen. Mechanism eligibility further restricts cohorts; Reset has 167 eligible improved samples. Preserve denominators and exclusion reasons in analysis.
