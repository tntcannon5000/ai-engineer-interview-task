- Read the README and looked through the repo to understand the assignment, what was already there and what might need fixing.
- Ran setup, switched to embedding model 2 after quota errors with the original model, rebuilt the database and checked search.

Order of work:
1. implement notes support (complete)
2. clean up repo issues (complete)
3. extended work, ie frontend, simulated Granola integration and improving search

1. Notes support:
   - Kept SQLite, with separate embeddings for resumes and notes. Means new notes can be added without re-embedding the resume, and the original text and timestamps stay available.
   - Added notes on candidate creation, adding notes to existing candidates and a repeatable import for notes.json. Started with the best resume/note similarity per candidate so notes could bring someone into the results without showing duplicate candidates.
   - Checked creation, adding notes and search with live Gemini calls. Imported all 119 supplied notes for the 100 candidates.

2. Clean up repo issues:
   - Removed destructive database administration from the API and moved it into scripts. Why: maintenance shouldn't be accessible through the recruiter UI.
   - Validate inputs and generate embeddings before writing; made rebuilds atomic and check for concurrent changes. Why: a failed rebuild should leave existing data intact.
   - Tightened request/import validation, normalised skills and locations, and made provider errors explicit. Why: avoid inconsistent filtering and partially saved records.
   - Record the embedding model, dimensions and format version, and reject incompatible vectors. Why: matching dimensions alone doesn't mean two sets of embeddings can be compared.
   - Kept messy notes, with diagnostics for placeholders and numeric-only entries. Why: make the issues visible without throwing away information before deciding how to classify it.
   - Added exact name/resume deduplication, also enforced in SQLite. Why: prevent repeated submissions and unnecessary model calls. Different resume versions aren't automatically merged.
   - Kept the sample resumes in their original format, without added IDs. Notes map through their original source positions, with fingerprint checks on repeat imports. Why: preserve the supplied data and avoid attaching notes to the wrong person.
   - Made setup repeatable and checked data integrity and note ownership. Why: running setup again should not duplicate imported data.

3. Extended work:
   - Added a simple single-file recruiter frontend for search, candidate details and notes.
   - Added drag/drop for TXT, Markdown, text-based PDF and DOCX. Extract text locally first, then use Pydantic AI structured output for an editable candidate draft. Reviewed fields don't need extracting again on save.
   - Left OCR for later rather than add another service. Would need page-level text-quality checks, time/cost limits and review of uncertain extraction.
   - Added simulated Granola recent transcripts to the add-information flow. Pick the candidate explicitly, keep source metadata and prevent repeat imports. Real authentication/sync remains separate work, as does broader candidate identity matching.
   - Improved retrieval to look for each requirement across resumes and notes. Used a soft minimum across mandatory requirements so one strong match doesn't outweigh several weak ones as easily.
   - Added one batched LLM assessment for a fixed shortlist of 12 candidates. Returns supported / contradicted / unknown judgments and source quotes; code checks the quotes and determines the order.
   - Capped requirements, evidence size and validation retries to bound cost and latency. Kept the original similarity search available for comparison. Checking a quote exists doesn't prove the judgment is correct.
   - Added repeatable evals with relevance metrics, stage timings, token usage and estimated cost. Checked the evidence display in the UI; evals don't change candidate data.
   - Compared nine draft queries: eight with known positives, one with a known contradiction. Recall@5 stayed at 100%; MRR@5 went from 0.875 to 0.938. The gain was one query's first relevant candidate moving from second to first.
   - Average time went from 0.35s to 4.20s. Estimated cost was $0.049 for the nine-query run, about 0.55 US cents each. One provider failure needed a retry; actual billing isn't measured and failed-request usage is incomplete.
   - Shelved broader experiments within the take-home time budget. Labels aren't exhaustive and the dataset is small; this doesn't establish better recall or prove the extra assessment is worth using in production.
   - Added candidate lookup by name and replaced the supported-count badges with a 0–4 relevance grade, calculated from the existing judgments without extra model calls. Mandatory contradictions score 0; unknown information is different from a contradiction, and preferences alone don't make someone incompatible. The eval figures above predate this grading change.
   - Expanded lookup into a larger alphabetical directory. Search-result profiles link back to it for adding notes/documents. Applied Popp's wordmark, typography and purple colour theme.
   - Profiled the wait: most of it was assessment. Added separate answer/thinking token accounting; sampled minimal-thinking calls had no separately accounted thinking tokens. Tried compact output, but results were inconsistent, so kept the original model contract.
   - Added simulated loading steps in the frontend, with no previous results left on screen. Results show as sostaron as they return. This is feedback during the wait, not real backend progress or a reduction in latency.
   - Checked lookup/edit handoff, loading and errors, database integrity and submission exclusions. 147 automated tests passed. Private preparation notes and generated reports stay local.

Next things to investigate:
- First, a larger recruiter-reviewed eval set: negation, information spread across documents, conflicting notes and unknowns. Compare retrieval and assessment separately to see what earns the extra cost and wait.
- Then follow the actual failures: better evidence selection/chunking for long documents, classification of low-value notes, model/prompt comparisons and shortlist tuning. Whole-document embeddings and truncated evidence can currently miss important details.
