# Third-Party Notices

## Qwen2.5 Model Components

This project uses derived data from the Qwen2.5 models developed by Alibaba Cloud:

- `data/tokenizer.json` — Tokenizer configuration from Qwen2.5-0.5B-Instruct
- `data/distilled_7b.npz` — Distilled anchor vectors derived from Qwen2.5-7B-Instruct embedding table
- `data/pca_256_proj.npz` — PCA projection matrix computed from Qwen2.5-7B-Instruct embedding table

**License:** Apache License 2.0

The Qwen2.5 models (0.5B, 1.5B, 7B variants) are licensed under the Apache License 2.0, which permits commercial use, modification, and distribution. The full license text is available at: https://www.apache.org/licenses/LICENSE-2.0

**Attribution:**
```
Qwen2.5 by Alibaba Cloud
https://github.com/QwenLM/Qwen2.5
Licensed under Apache 2.0
```

## Font Reference

The engine references system fonts for bitmap density calculation:
- Windows: `C:/Windows/Fonts/simsun.ttc` (SimSun / 宋体)

This project does not redistribute font files. Users on Linux/macOS need to provide their own CJK font and update the path in `engine/engine_v9.py`.

## Python Dependencies

See `requirements.txt` for the list of Python package dependencies. Each dependency has its own license terms.
