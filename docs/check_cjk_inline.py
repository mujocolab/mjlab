#!/usr/bin/env python3
"""CJK 行内标记残留检测与修复器。

docutils 要求粗体 **x**、斜体 *x*、行内代码 ``x`` 与角色 :r:`x` 的
开始标记前不能紧邻汉字,结束标记后不能紧跟汉字或全角括号;否则整个
标记不渲染,星号与反引号字面残留。

检测:用 docutils 解析 docs/zh 树的每个 rst,收集行内标记失败类 warning。
修复:对失败行做行内标记成对扫描,只在真正的标记边界插半角空格
(不会把空格插进标记内部——这是 v1 正则方案的教训,勿回退)。

用法:
  uv run --group docs python docs/check_cjk_inline.py           # 检测
  uv run --group docs python docs/check_cjk_inline.py --fix     # 检测+修复
"""

import re
import sys
from pathlib import Path

from docutils.core import publish_doctree

REPO = Path(__file__).resolve().parent.parent
ZH = REPO / "docs" / "zh"

FAIL_PATTERNS = (
  "start-string without end-string",
  "end-string without",
  "Inline interpreted text or phrase reference start-string",
)

HAN = r"[\u4e00-\u9fff]"
BAD_AFTER = set("（）〈〉《》【】『』「」") | set(
  chr(c) for c in range(0x4E00, 0xA000))

# 行内标记成对扫描:角色 / 双反引号 / 粗体 / 斜体(alternation 有序)
MARKUP = re.compile(
  r"(:[a-zA-Z0-9_+-]+:`[^`]+`"   # :role:`text`
  r"|``[^`]+``"                   # ``literal``
  r"|\*\*[^*\n]+\*\*"             # **bold**
  r"|\*[^*\n]+\*)"                # *emphasis*
)


def detect_file(path: Path) -> list:
  """返回 [(行号, 消息), ...]。docutils 把 warning 嵌在 doctree 的
  system_message 节点里,而不是写入 warning_stream。"""
  doctree = publish_doctree(
    path.read_text(encoding="utf-8"),
    source_path=str(path),
    settings_overrides={"report_level": 5, "halt_level": 5},
  )
  out = []
  for node in doctree.findall(lambda n: n.tagname == "system_message"):
    msg = node.astext()
    if any(p in msg for p in FAIL_PATTERNS):
      out.append((int(node.get("line") or 0), msg.strip()))
  return out


def fix_line(line: str) -> str:
  """成对扫描行内标记,在越界边界插半角空格。

  行首 rst bullet(`* `)先剥离再扫描,避免 bullet 星号与标记星号
  混淆成错误 span。
  """
  bullet = ""
  body = line
  m = re.match(r"^(\s*\*\s)", line)
  if m:
    bullet, body = m.group(1), line[m.end():]
  insertions = []
  for mm in MARKUP.finditer(body):
    s, e = mm.span()
    if re.match(HAN, body[s - 1]) if s > 0 else False:
      insertions.append(s)
    if e < len(body) and body[e] in BAD_AFTER:
      insertions.append(e)
  for pos in sorted(set(insertions), reverse=True):
    body = body[:pos] + " " + body[pos:]
  return bullet + body


def block_range(lines: list, lineno: int) -> tuple:
  """返回报告行所属段落/块的 (起,止) 行号区间 [起,止)。

  向上回溯到空行,向下延伸到空行;表格等缩进块的连续行都会覆盖。
  """
  top = lineno - 1
  while top > 0 and lines[top - 1].strip():
    top -= 1
  bot = lineno
  while bot < len(lines) and lines[bot].strip():
    bot += 1
  return top, bot


def main() -> int:
  fix = "--fix" in sys.argv
  files = sorted(ZH.rglob("*.rst")) + [ZH / "index.rst"]
  total, report = 0, []
  for path in files:
    if not path.exists():
      continue
    hits = detect_file(path)
    if not hits:
      continue
    total += len(hits)
    report += [f"{path.relative_to(REPO)}:{ln}: {msg[:70]}" for ln, msg in hits]
    if fix:
      lines = path.read_text(encoding="utf-8").splitlines()
      fixed_blocks = set()
      for ln, _ in hits:
        top, bot = block_range(lines, ln)
        if (id(lines), top, bot) in fixed_blocks:
          continue
        fixed_blocks.add((id(lines), top, bot))
        for i in range(top, bot):
          lines[i] = fix_line(lines[i])
      path.write_text("\n".join(lines) + "\n", encoding="utf-8")

  verb = "修复后复检" if fix else "检测"
  print(f"[{verb}] 行内标记残留 {total} 处")
  for r in report:
    print(" -", r)
  return 1 if total and not fix else 0


if __name__ == "__main__":
  sys.exit(main())
