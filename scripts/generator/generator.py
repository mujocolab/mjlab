import shutil
from pathlib import Path
import re

def copy_dir(src: Path, dst: Path, *, overwrite: bool = True):
    """Copy a directory and all its contents to dst.
    
    Args:
        src: Path to source directory.
        dst: Path to target directory.
        overwrite: If True, delete dst first if it exists.
    """
    if not src.is_dir():
        raise ValueError(f"Source is not a directory: {src}")

    if dst.exists():
        if overwrite:
            shutil.rmtree(dst)
        else:
            raise FileExistsError(f"Destination already exists: {dst}")

    shutil.copytree(src, dst)
    print(f"[ok] copied {src} -> {dst}")

def to_import_name(dist_name: str) -> str:
    name = dist_name.lower()
    name = re.sub(r"[^a-z0-9_]", "_", name)
    if name[0].isdigit():
        name = "_" + name
    return name


# TODO: sh function uv make + uv add mjlab

def write_or_copy(path: Path, content: str | None = None, src: Path | None = None, *, force=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        print(f"[skip] {path} exists")
        return

    if src is not None:
        shutil.copy(src, path)
        print(f"[ok] copied {src} -> {path}")
    else:
        path.write_text(content or "", encoding="utf-8")
        print(f"[ok] wrote {path}")


from typing import Callable


def replace_line(path: Path, line_no: int, new_content: str | None = None):
    """Replace or remove a specific line in a file.
    
    Args:
        path: Path to the file.
        line_no: Line number (1-based).
        new_content: Replacement string. If None, the line is removed.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    
    if not (1 <= line_no <= len(lines)):
        raise IndexError(f"Line {line_no} out of range (file has {len(lines)} lines)")
    
    if new_content is None:
        print(f"[ok] removed line {line_no}: {lines[line_no-1]!r}")
        del lines[line_no-1]
    else:
        print(f"[ok] replaced line {line_no}: {lines[line_no-1]!r} -> {new_content!r}")
        lines[line_no-1] = new_content
    
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main():
    # TODO: add arguments

    project_name = "example-pkg"
    import_name = to_import_name(project_name)

    # path vars
    mjlab_root = Path(__file__).resolve().parent.parent.parent
    mjlab_content = mjlab_root / "src" / "mjlab"

    templates = mjlab_root / "scripts" / "generator" / "templates"
    core = templates / "core"

    project_root = mjlab_root.parent / project_name
    project_content = project_root / "src" / import_name

    # copy/modify core files
    write_or_copy(
        path = project_root / "README.md", 
        content = "aaa"
    )
    write_or_copy(
        path = project_content / "__init__.py", 
        src = core / "__init__.py"
    )

    # robots files
    write_or_copy(
        path = project_content / "robots" / "__init__.py",
        src = templates / "robots" / "__init__.py"
    )
    copy_dir(
        src = mjlab_content / "asset_zoo" / "robots" / "unitree_go1", 
        dst = project_content / "robots" / "unitree_go1"
    )

    # task files
    write_or_copy(path = project_content / "tasks" / "__init__.py")
    copy_dir(
        src = mjlab_content / "tasks" / "locomotion" / "velocity" / "config" / "go1", 
        dst = project_content / "tasks" / "go1_locomotion"
    )

    # scripts
    write_or_copy(
        path = project_root / "scripts" / "list_envs.py",
        src = mjlab_root / "scripts" / "list_envs.py"
    )
    copy_dir(
        src = mjlab_root / "scripts" / "velocity" / "rl", 
        dst = project_root / "scripts" / "rl"
    )

    #modifying scriptsss
    replace_line(
        project_root / "scripts" / "list_envs.py", 
        6, 
        f"import {import_name}.tasks  # noqa: F401 to register environments"
    )

    # modifying core
    replace_line(
        project_content / "__init__.py", 
        3, 
        f"{import_name}_SRC_PATH: Path = Path(__file__).parent"
    )


if __name__ == "__main__":
    main()