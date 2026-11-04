# Project change documentation

- Every change to code, configuration, templates, notebooks, data contracts, or runtime behavior must update `docs/NHAT_KY_THAY_DOI_VI.md` (changes and results) and, when the architecture changes, `docs/KIEN_TRUC_CHI_TIET_VI.md` in the same commit.
- The documentation entry must state what changed, why it changed, compatibility or safety implications, and the validation performed.
- For IPN symbolic/template work, append the dated entry to `docs/NHAT_KY_THAY_DOI_VI.md`; the former per-topic notes were merged into the two files above and remain in git history.
- Do not stage generated result notebooks or experiment artifacts unless the user explicitly asks to version them.
