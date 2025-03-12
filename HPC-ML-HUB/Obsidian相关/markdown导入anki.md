---
tags:
  - 导入
  - 闪卡
  - anki
aliases:
  - 导入闪卡
  - mdanki
  - anki
---

###  可导入anki的markdown的书写格式
```
## What's the Markdown?

Markdown is a lightweight markup language with plain-text-formatting syntax.
Its design allows it to be converted to many output formats,
but the original tool by the same name only supports HTML.

## Who created Markdown?

John Gruber created the Markdown language in 2004 in collaboration with
Aaron Swartz on the syntax.


## if you want to make the question in multiple lines.

% 

use `%` symbol for splitting front and back sides
```

### 格式转化

Convert a single markdown file:

```shell
mdanki library.md anki.apkg
```

Convert files from directory recursively:

```shell
mdanki ./documents/library ./documents/anki.apkg
```

Using all available options:

```shell
mdanki library.md anki.apkg --deck Library --config config.json
```

Import just generated `.apkg` file to Anki ("File" - "Import").