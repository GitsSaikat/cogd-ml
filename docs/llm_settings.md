# Language model settings

Reproduces S1 Appendix. Nothing in the main paper depends on these runs.

Base model `Qwen/Qwen2.5-1.5B-Instruct`, seed 42, identical across all three
paradigms so that only the adaptation method varies.

| | Setting |
|---|---|
| Quantisation | 4-bit NF4, double quantisation, bfloat16 compute |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Epochs / LR / batch | 3 / 2e-4 / 4 with 4-step accumulation |
| Retriever | all-MiniLM-L6-v2, top-5 nearest training examples |
| Decoding | greedy, max 8 new tokens |

Prompts are in `prompts/`. Zero-shot and RAG share `system.txt`; RAG wraps the
serialised case in `rag.txt`.

All three paradigms scored below the classical classifiers. QLoRA was the
strongest at MCC 0.52, retrieval-augmented generation effectively failed on the
minority class at MCC 0.01.
