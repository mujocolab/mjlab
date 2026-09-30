#!/usr/bin/env python3
"""中英双语文档一致性校验脚本。

校验项:
  1. 源文件集合 parity —— docs/ 与 docs/zh/ 的 rst 文件一一对应(changelog.rst 允许内容相同)
  2. 章节结构 parity —— 每个文件的章节下划线层级计数一致
  3. 代码块 parity —— code-block/confirm 类指令的代码内容逐字节一致
  4. 构建产物校验(可选, --build) —— 每个页面的语言切换链接目标存在

用法:
  python3 docs/check_bilingual.py                # 校验 1-3(源码级)
  python3 docs/check_bilingual.py --build docs/_build  # 追加校验 4(需先构建)
"""

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EN_ROOT = REPO / "docs"
ZH_ROOT = REPO / "docs" / "zh"
# 中文站页面与英文站 pagename 一致;校验时把 docs/zh 视作"源树"。
# 英文树的 index.rst 对应 docs/index.rst, 中文树为 docs/zh/index.rst。

UNDERLINE_CHARS = set("=-~^\"'+#*_")

# 已知章节结构豁免:英文源存在 rst 瑕疵(标题长于下划线导致章节不被
# 识别),中文版书写为合法章节,属预期差异而非翻译遗漏。
KNOWN_SECTION_EXCEPTIONS = {
  "faq.rst": "英文版 'What is flat patch sampling...' 标题比其下划线长 1 字符,"
  "docutils 不识别为章节;中文版为合法章节。",
}


def relative_rst_files(root: Path) -> dict:
  """返回 {相对键: 文件绝对路径}。

  两棵树结构同构: <root>/index.rst + <root>/source/**.rst,
  键统一为 "index.rst" 或相对 source 的路径,便于一一比对。
  """
  files = {"index.rst": root / "index.rst"}
  for p in sorted((root / "source").rglob("*.rst")):
    files[str(p.relative_to(root / "source"))] = p
  return files


def section_counts(text: str) -> dict:
  """统计 rst 章节下划线使用次数: {下划线字符: 数量}。"""
  lines = text.splitlines()
  counts = {}
  for i in range(1, len(lines)):
    line = lines[i].strip()
    prev = lines[i - 1].strip()
    if (
      line
      and len(set(line)) == 1
      and line[0] in UNDERLINE_CHARS
      and len(line) >= len(prev)
      and prev
      and not prev.startswith((".. ", ":"))
    ):
      counts[line[0]] = counts.get(line[0], 0) + 1
  return counts


CODE_DIRECTIVE = re.compile(r"^\.\. (code-block|jupyter-execute)::", re.M)


def code_blocks(text: str) -> list:
  """抽取 code-block 指令的缩进代码体(逐字节)。"""
  lines = text.splitlines()
  blocks, i = [], 0
  while i < len(lines):
    if CODE_DIRECTIVE.match(lines[i]):
      i += 1
      # 跳过指令选项行(以 : 开头)
      while i < len(lines) and lines[i].strip().startswith(":"):
        i += 1
      block = []
      while i < len(lines) and (
        lines[i].startswith((" ", "\t")) or not lines[i].strip()
      ):
        block.append(lines[i])
        i += 1
      blocks.append("\n".join(block).rstrip())
    else:
      i += 1
  return blocks


def main() -> int:
  parser = argparse.ArgumentParser(description="中英双语文档一致性校验")
  parser.add_argument("--build", help="已构建的 _build 目录,启用切换链接校验")
  args = parser.parse_args()

  en_files = relative_rst_files(EN_ROOT)
  zh_files = relative_rst_files(ZH_ROOT)

  failures = []

  # 1. 文件集合 parity
  only_en = sorted(set(en_files) - set(zh_files))
  only_zh = sorted(set(zh_files) - set(en_files))
  if only_en:
    failures.append(f"中文树缺失文件: {only_en}")
  if only_zh:
    failures.append(f"中文树多余文件: {only_zh}")
  print(
    f"[1] 文件集合: 英文 {len(en_files)} 个, 中文 {len(zh_files)} 个, "
    f"缺失 {len(only_en)}, 多余 {len(only_zh)}"
  )

  # 2/3. 逐文件章节结构与代码块 parity
  sec_bad = code_bad = 0
  for key in sorted(set(en_files) & set(zh_files)):
    en_text = en_files[key].read_text(encoding="utf-8")
    zh_text = zh_files[key].read_text(encoding="utf-8")
    if section_counts(en_text) != section_counts(zh_text):
      if key in KNOWN_SECTION_EXCEPTIONS:
        print(f"[2] 章节结构(已知豁免): {key} - {KNOWN_SECTION_EXCEPTIONS[key]}")
      else:
        sec_bad += 1
        failures.append(
          f"章节结构不一致: {key} "
          f"英文={section_counts(en_text)} 中文={section_counts(zh_text)}"
        )
    if code_blocks(en_text) != code_blocks(zh_text):
      code_bad += 1
      failures.append(f"代码块内容不一致: {key}")
  print(f"[2] 章节结构: 不一致 {sec_bad} 个")
  print(f"[3] 代码块: 不一致 {code_bad} 个")

  # 4. 构建产物切换链接校验
  if args.build:
    build = Path(args.build)
    checked = broken = 0
    for html in build.rglob("*.html"):
      rel = html.relative_to(build)
      text = html.read_text(encoding="utf-8", errors="ignore")
      m = re.search(r'class="doc-lang-switch">\s*<a href="([^"]+)"', text)
      if not m:
        continue
      checked += 1
      target = (html.parent / m.group(1)).resolve()
      if not target.is_file():
        broken += 1
        failures.append(f"切换链接失效: {rel} -> {m.group(1)}")
    print(f"[4] 切换链接: 检查 {checked} 个, 失效 {broken} 个")

  # 5. 渲染星号残留断言(仅 zh 站;剔除代码/pre 后正文不得含 ** 或紧邻汉字的 *)
  if args.build:
    build = Path(args.build)
    star_bad, star_checked = 0, 0
    for html in (build / "zh").rglob("*.html"):
      raw = html.read_text(encoding="utf-8", errors="ignore")
      raw = re.sub(r"<(pre|code|script|style)[\s\S]*?</\1>", "", raw)
      raw = re.sub(r"<[^>]+>", "", raw)  # 剥标签,只看正文文本
      raw = re.sub(r"\*{1,2}(?:args|kwargs)", "", raw)  # 排除 Python 签名
      star_checked += 1
      if "**" in raw or re.search(r"\*[\u4e00-\u9fff]|[\u4e00-\u9fff（]\*", raw):
        star_bad += 1
        failures.append(f"星号残留: {html.relative_to(build)}")
    print(f"[5] 星号残留: 检查 {star_checked} 页, 异常 {star_bad} 页")

  if failures:
    print("\n=== 校验失败 ===")
    for f in failures:
      print(" -", f)
    return 1
  print("\n全部校验通过")
  return 0


if __name__ == "__main__":
  sys.exit(main())
