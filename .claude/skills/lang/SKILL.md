---
name: lang
description: Switch Alpha Trader output language (en/zh/zh-tw/es/ja/ko)
user-invocable: true
---

# Switch Output Language

## Usage
- `/lang` — show current language and options
- `/lang zh` — switch to Simplified Chinese

## Available codes
| Code | Language |
|------|----------|
| en | English (default) |
| zh | 简体中文 |
| zh-tw | 繁體中文 |
| es | Español |
| ja | 日本語 |
| ko | 한국어 |

## How to execute

If the user provides a language code (e.g. `/lang zh`), run:
```bash
cd /Users/xxzhou/OSS/alpha-trader && ./at lang CODE_HERE
```

Wait — there's no `lang` CLI command yet. Instead run:
```bash
cd /Users/xxzhou/OSS/alpha-trader && source .venv/bin/activate && python -c "
from src.config import set_language, LANGUAGES
result = set_language('CODE_HERE')
print(f'Set to: {result}' if result else 'Invalid. Options: ' + ', '.join(f'{k} ({v})' for k,v in LANGUAGES.items()))
"
```

After switching, confirm in the chosen language.
