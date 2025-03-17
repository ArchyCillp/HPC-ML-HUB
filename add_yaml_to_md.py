import os
import glob

# 遍历所有 Markdown 文件
for filepath in glob.glob("**/*.md", recursive=True):
    with open(filepath, "r+", encoding="utf-8") as f:
        content = f.read()
        if not content.startswith("---"):
            # 添加默认文件头
            filename = os.path.basename(filepath).replace(".md", "")
            header = f"---\ntitle: {filename}\n---\n\n"
            f.seek(0)
            f.write(header + content)
