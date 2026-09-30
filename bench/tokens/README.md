# Evaluation token sequences

`<scenario>.prompt` holds the chat-templated prompt token ids for Qwen3.8-Flash-Next (Qwen tokenizer,
vocabulary 248,320). `<scenario>.replay` holds the continuation, one id per line. The continuation
was generated once, greedily, by the original QwFN (2d6837f). Every engine, llama.cpp included, is
scored teacher-forced on exactly these ids.

| Scenario | Source | Tokens scored |
|---|---|---|
| code | Python LRU cache and tests, thinking off | 800 |
| agent | review and refactor of `agent_file.txt` (a C++ header excerpt), thinking off | 800 |
| short | three-sentence explanation, thinking on | 300 |
| reasoning | constrained optimisation word problem, thinking on | 600 |
| longctx | `src/qwfn_expert_cache.cpp` as of commit 7384648 (~21K tokens; the token file is authoritative) plus a grounded question, thinking off | 300 |

The prompt texts are in `bench/make_tokens.sh`. Regenerating the files changes the continuation
and invalidates comparisons with the published evidence.
