import logging
import re
import shutil
import tempfile
import uuid
import weakref
from pathlib import Path
from typing import Optional

from gerrytools._ipython import in_jupyter_kernel as _in_jupyter_kernel
from gerrytools.latex import _render
from gerrytools.latex._colors import (
    _LATEX_COLOR_NAMES,
    LatexColorSpec,
    to_latex_xcolor_or_html_spec,
)
from gerrytools.logging import get_logger
from gerrytools.typing import Color

logger = get_logger(__name__)

_USEPACKAGE_WITH_OPTIONS_RE = re.compile(r"\\usepackage\[[^\]]*\]\{(?P<name>[^}]+)\}")
"""Matches the ``\\usepackage[options]{name}`` entries built by ``add_package_with_options``."""
_COLOR_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*")


class TexDocument:
    """Class for creating and previewing LaTeX documents.

    Allows adding packages, commands, and colors, and can render the LaTeX content to a PNG for
    display in Jupyter or a Qt window.

    Attributes:
        body_string (str): The LaTeX document body content.
        package_list (list[str]): List of LaTeX packages to include.
        extra_package_commands (list[str]): Additional LaTeX package commands.
        command_list (list[str]): List of custom LaTeX commands.
        color_dict (dict[str, LatexColorSpec]): Dictionary of color definitions.
        engine_preference_order (tuple[str, ...]): Preferred order of TeX engines to use.
    """

    def __init__(self) -> None:
        self._uuid = uuid.uuid4().hex
        # The on-disk workspace is created lazily by `_workdir` on first compile/preview/save:
        # every table and plot owns a document, so construction must not touch the filesystem.
        self._workdir_path: Optional[Path] = None
        self._finalizer: Optional[weakref.finalize] = None
        self.body_string: str = ""
        # The preamble stays minimal: features register the packages they need, and `preamble`
        # scans the body and commands for macros that imply the rest.
        self.package_list: list[str] = []
        self.extra_package_commands: list[str] = []
        self.command_list: list[str] = []
        self.color_dict: dict[str, LatexColorSpec] = {}
        self.engine_preference_order = ("tectonic", "pdflatex", "xelatex", "lualatex")
        self.compile_passes: int = 1
        """Number of LaTeX passes per compile. Packages that persist node
        positions in the aux file (e.g. nicematrix) need 2."""

    @property
    def _workdir(self) -> Path:
        """Temporary workspace for compile artifacts, created on first use.

        The cleanup finalizer attaches when the directory is created, not when the document is
        constructed, so documents that never compile leave nothing behind.
        """
        if self._workdir_path is None:
            self._workdir_path = Path(tempfile.mkdtemp(prefix="latex-preview-"))
            self._finalizer = weakref.finalize(
                self,
                shutil.rmtree,
                self._workdir_path,
                True,  # ignore_errors
            )
        return self._workdir_path

    @property
    def _tex_path(self) -> Path:
        return self._workdir / f"{self._uuid}.tex"

    @property
    def _pdf_path(self) -> Path:
        return self._workdir / f"{self._uuid}.pdf"

    @property
    def _png_path(self) -> Path:
        return self._workdir / f"{self._uuid}.png"

    def __repr__(self) -> str:  # pragma: no cover
        return self._tex_doc_string()

    def __str__(self) -> str:  # pragma: no cover
        return self._tex_doc_string()

    def _optioned_package_names(self) -> set[str]:
        """Get the names of packages already loaded with options.

        Returns:
            set[str]: Package names appearing in ``extra_package_commands`` as
                ``\\usepackage[...]{name}`` entries.
        """
        names: set[str] = set()
        for command in self.extra_package_commands:
            match = _USEPACKAGE_WITH_OPTIONS_RE.fullmatch(command)
            if match is not None:
                names.add(match.group("name"))
        return names

    def add_packages(self, packages: str | list[str]) -> None:
        """Adds one or more LaTeX packages to the package list.

        A package already registered with options via :meth:`add_package_with_options` is skipped
        since loading the same package twice with different options is a LaTeX "option clash" error.

        Examples:
            tex_preview.add_packages("geometry")
            tex_preview.add_packages(["geometry", "hyperref"])

        Args:
            packages (str | list[str]): A single package name or a list of package names to add.

        Returns:
            None
        """
        if isinstance(packages, str):
            package_list = [packages]
        else:
            package_list = packages
        optioned_names = self._optioned_package_names()
        for package_name in package_list:
            if package_name in optioned_names:
                continue
            if package_name not in self.package_list:
                self.package_list.append(package_name)

    def add_package_with_options(self, package_name: str, options: str | list[str]) -> None:
        """Adds a LaTeX package with options to the extra package commands.

        The optioned form supersedes any plain ``\\usepackage{name}`` entry, and re-registering the
        same package replaces its previous options.

        Examples:
            tex_preview.add_package_with_options("geometry", "margin=1in")
            tex_preview.add_package_with_options("hyperref", ["colorlinks", "linkcolor=blue"])

        Args:
            package_name (str): The name of the LaTeX package to add.
            options (str | list[str]): A single option string or a list of option strings.

        Returns:
            None
        """
        options_list = [options] if isinstance(options, str) else options
        options_str = ",".join(options_list)
        pkg_cmd = rf"\usepackage[{options_str}]{{{package_name}}}"
        if package_name in self.package_list:
            self.package_list.remove(package_name)
        for index, command in enumerate(self.extra_package_commands):
            match = _USEPACKAGE_WITH_OPTIONS_RE.fullmatch(command)
            if match is not None and match.group("name") == package_name:
                self.extra_package_commands[index] = pkg_cmd
                return
        self.extra_package_commands.append(pkg_cmd)

    def add_command(self, command: str) -> None:
        r"""Adds a custom LaTeX command to the document.

        Examples:
            tex_preview.add_command(r"\newcommand{\R}{\mathbb{R}}")

        Args:
            command (str): The LaTeX command to add.

        Returns:
            None
        """
        self.command_list.append(command)

    def add_color(self, color_name: str, color: Color) -> None:
        """Adds a custom color definition to the document.

        Examples:
            tex_preview.add_color("myblue", (0.0, 0.0, 1.0))
            tex_preview.add_color("brand", "tab:blue")
            tex_preview.add_color("accent", "red!60!black")

        Args:
            color_name (str): The name of the color to define.
            color (Color): xcolor expression, HEX string, parseable named color, or RGB
                tuple. RGB tuple values must all be in ``[0, 1]`` or all be in
                ``[0, 255]``.

        Returns:
            None

        Raises:
            ValueError: If ``color_name`` is not a safe LaTeX identifier or ``color`` is not a
                valid xcolor expression/HEX/RGB value.
        """
        if not isinstance(color_name, str) or _COLOR_NAME_RE.fullmatch(color_name) is None:
            raise ValueError(
                "Color name must start with an ASCII letter and contain only ASCII letters, "
                "digits, and '-'."
            )
        if isinstance(color, str) and color.strip().lower() == "none":
            raise ValueError("Color value 'none' cannot be registered in the document.")
        try:
            color_type, color_value = to_latex_xcolor_or_html_spec(color)
        except ValueError as exc:
            raise ValueError(
                "Color must be an xcolor expression, HEX string, parseable color name, "
                "or RGB tuple of length 3 with components in [0, 1] or [0, 255]."
            ) from exc
        if color_type == "NAME":
            assert isinstance(color_value, str)
            self.color_dict[color_name] = ("NAME", color_value)
            return
        if color_type == "HTML":
            assert isinstance(color_value, str)
            self.color_dict[color_name] = ("HTML", color_value)
            return
        if color_type == "rgb":
            assert isinstance(color_value, tuple)
            self.color_dict[color_name] = (
                "rgb",
                (round(color_value[0], 2), round(color_value[1], 2), round(color_value[2], 2)),
            )
            return
        assert isinstance(color_value, tuple)
        self.color_dict[color_name] = (
            "RGB",
            (int(color_value[0]), int(color_value[1]), int(color_value[2])),
        )

    # Macros whose presence in the body or custom commands implies a package. The scan keeps
    # the default preamble minimal while making pasted `to_tex()` output compile.
    _MACRO_PACKAGE_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
        (r"\\(?:top|mid|bottom)rule|\\cmidrule|\\specialrule", ("booktabs",)),
        (r"\\rowcolor|\\cellcolor|\\arrayrulecolor", ("colortbl",)),
        (r"\\fpeval", ("xfp",)),
        (r"\\num[\[{]|\\tablenum", ("siunitx",)),
        (r"\\includegraphics", ("graphicx",)),
        (r"\\multirow", ("multirow",)),
        (r">\{", ("array",)),
    )

    _COLOR_MACRO_RE = re.compile(
        r"\\(?:rowcolor|cellcolor|arrayrulecolor|textcolor|colorbox|colorlet|definecolor|"
        r"color(?![a-zA-Z]))"
    )

    _PACKAGE_ORDER: tuple[str, ...] = (
        "amsmath",
        "amssymb",
        "graphicx",
        "array",
        "dcolumn",
        "multirow",
        "booktabs",
        "xcolor",
        "latexcolors",
        "colortbl",
        "siunitx",
        "xfp",
        "tikz",
        "nicematrix",
    )

    # xcolor's base names double as latexcolors names; they never require latexcolors.
    _XCOLOR_BASE_NAMES = frozenset(
        "black blue brown cyan darkgray gray green lightgray lime magenta olive orange "
        "pink purple red teal violet white yellow".split()
    )
    _LATEX_COLOR_TOKEN_RE = re.compile(
        f"[{re.escape(''.join(sorted(set().union(*_LATEX_COLOR_NAMES))))}]{{2,}}"
    )
    _COLOR_VALUE_RE = re.compile(
        r"\\(?:rowcolor|cellcolor|arrayrulecolor|textcolor|colorbox|color)"
        r"\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}"
        r"|\\colorlet\s*\{[^{}]*\}\s*\{([^{}]+)\}"
        r"|(?:^|[,\[])\s*(?:fill|draw|color|text)\s*="
        r"\s*(?:\{([^{}]+)\}|([^,\]\s]+))",
        re.MULTILINE,
    )

    def _uses_latexcolors_name(self) -> bool:
        """Whether any referenced color name needs the ``latexcolors`` package.

        Body text is restricted to color-valued macro arguments and TikZ options so ordinary prose
        cannot add a package. Registered commands are generated LaTeX code and may build color
        expressions indirectly, so their complete source is scanned.

        Returns:
            bool: Whether a latexcolors-only color name is referenced.
        """
        known_names = _LATEX_COLOR_NAMES - self._XCOLOR_BASE_NAMES
        candidates: set[str] = set()
        for match in self._COLOR_VALUE_RE.finditer(self.body_string):
            value = next(group for group in match.groups() if group is not None)
            candidates.update(self._LATEX_COLOR_TOKEN_RE.findall(value))
        candidates.update(self._LATEX_COLOR_TOKEN_RE.findall("\n".join(self.command_list)))
        for _, (color_type, color_value) in self.color_dict.items():
            if color_type == "NAME" and isinstance(color_value, str):
                candidates.update(self._LATEX_COLOR_TOKEN_RE.findall(color_value))
        return bool(candidates & known_names)

    def _scanned_packages(self) -> set[str]:
        """Packages implied by macros in the body and registered commands."""
        haystack = "\n".join([self.body_string, *self.command_list])
        found: set[str] = set()
        for pattern, packages in self._MACRO_PACKAGE_HINTS:
            if re.search(pattern, haystack):
                found.update(packages)
        # latexcolors names can also appear in bare TikZ options such as fill=cadmiumgreen.
        if self._uses_latexcolors_name():
            found.add("latexcolors")
        elif self._COLOR_MACRO_RE.search(haystack) or self.color_dict:
            found.add("xcolor")
        return found

    def _resolved_package_list(self) -> list[str]:
        """Explicit registrations plus scanned requirements, in a stable order."""
        wanted = set(self.package_list) | self._scanned_packages()
        wanted -= self._optioned_package_names()
        if "latexcolors" in wanted:
            wanted.discard("xcolor")
        ordered = [pkg for pkg in self._PACKAGE_ORDER if pkg in wanted]
        ordered += [pkg for pkg in self.package_list if pkg not in ordered and pkg in wanted]
        return ordered

    @property
    def preamble(self) -> str:
        """Build the LaTeX preamble for this document.

        Returns:
            str: LaTeX preamble string including document class, packages, and color
                definitions.

        Raises:
            ValueError: If a registered package option or color definition is invalid.
        """
        lines = [r"\documentclass[border=2pt]{standalone}"]
        # latexcolors and colortbl can load xcolor themselves, so its optioned form must win the
        # race. Other optioned packages retain their registration order after scanned packages.
        early_commands = [
            command
            for command in self.extra_package_commands
            if (match := _USEPACKAGE_WITH_OPTIONS_RE.fullmatch(command)) is not None
            and match.group("name") == "xcolor"
        ]
        lines.extend(early_commands)
        packages = self._resolved_package_list()
        if packages:
            lines.append(r"\usepackage{" + ", ".join(packages) + "}")
        lines.extend(
            command for command in self.extra_package_commands if command not in early_commands
        )
        for color_name, (color_type, color_val) in self.color_dict.items():
            match color_type:
                case "NAME":
                    lines.append(rf"\colorlet{{{color_name}}}{{{color_val}}}")
                case "rgb":
                    assert isinstance(color_val, tuple), "Invalid color value for rgb."
                    lines.append(
                        rf"\definecolor{{{color_name}}}{{rgb}}{{{color_val[0]:0.2f},"
                        rf"{color_val[1]:0.2f},{color_val[2]:0.2f}}}"
                    )
                case "RGB":
                    assert isinstance(color_val, tuple), "Invalid color value for RGB."
                    lines.append(
                        rf"\definecolor{{{color_name}}}{{RGB}}{{{int(round(color_val[0]))},"
                        rf"{int(round(color_val[1]))},{int(round(color_val[2]))}}}"
                    )
                case "HTML":
                    lines.append(rf"\definecolor{{{color_name}}}{{HTML}}{{{color_val}}}")
                case _:  # pragma: no cover
                    raise ValueError(
                        f"Unsupported color type: {color_type} found in color dictionary. "
                        "Only 'NAME', 'rgb', 'RGB', and 'HTML' are supported."
                    )
        return "\n".join(lines)

    def to_tex(self) -> str:
        """The complete, compilable LaTeX document source.

        The returned string is a standalone document (preamble, commands, body) that can be pasted
        directly into a ``.tex`` file and compiled, or copied piecewise into an existing report.

        Returns:
            str: The full document source.
        """
        return self._tex_doc_string()

    def _tex_doc_string(self) -> str:
        """Generates the complete LaTeX document string."""
        lines = [self.preamble]
        lines.extend(self.command_list)
        lines += [r"\begin{document}", self.body_string, r"\end{document}"]
        output_string = "\n".join(lines).lstrip()
        logger.log(logging.DEBUG, "Generated LaTeX document:\n%s", output_string, stacklevel=2)
        return output_string

    def _compile_pdf(self, preferred_engine: Optional[str] = None) -> None:
        """Compile the current document body to a PDF in the working directory.

        Args:
            preferred_engine (Optional[str], optional): Preferred TeX engine name. If None, uses
                the first available engine from ``engine_preference_order``. Defaults to ``None``.

        Returns:
            None

        Raises:
            RuntimeError: If no TeX engine is found or LaTeX compilation fails.
        """
        _render._compile_pdf(
            self._tex_doc_string(),
            tex_path=self._tex_path,
            pdf_path=self._pdf_path,
            workdir=self._workdir,
            engine_preference_order=self.engine_preference_order,
            compile_passes=self.compile_passes,
            preferred_engine=preferred_engine,
        )

    def _render_to_temp_png(self, preferred_engine: Optional[str] = None, dpi: int = 250) -> None:
        """Render the current document body to a temporary PNG file.

        Args:
            preferred_engine (Optional[str], optional): Preferred TeX engine name. If None,
                uses the first available engine from ``engine_preference_order``. Defaults to
                ``None``.
            dpi (int, optional): PNG render resolution in dots-per-inch. Defaults to ``250``.

        Returns:
            None

        Raises:
            RuntimeError: If no TeX engine is found, LaTeX compilation fails, or
                no PDF renderer is available.
        """
        self._compile_pdf(preferred_engine)
        _render._render_pdf_to_png(self._pdf_path, self._png_path, dpi=dpi)

    def preview(self) -> None:  # pragma: no cover
        """Displays the rendered LaTeX document as a PNG image."""
        self._render_to_temp_png()
        if _in_jupyter_kernel():
            _render._show_png_jupyter(self._png_path)
        else:
            _render._show_png_qt(self._png_path, title="LaTeX Preview", max_size=(1200, 800))

    def save_pdf(self, filepath: str | Path) -> None:
        """Saves the rendered LaTeX document as a PDF file.

        Args:
            filepath (str | Path): The file path to save the PDF to.

        Returns:
            None

        Raises:
            TypeError: If ``filepath`` is not a string or Path.
            ValueError: If ``filepath`` does not end in ``.pdf``.
            FileNotFoundError: If the destination directory does not exist.
            RuntimeError: If LaTeX compilation fails or no TeX engine is available.
        """
        if not isinstance(filepath, (str, Path)):
            raise TypeError("Path must be a string or Path object.")

        if not str(filepath).endswith(".pdf"):
            raise ValueError("File extension must be '.pdf'")

        full_path = Path(filepath).resolve()

        if not full_path.parent.exists():
            raise FileNotFoundError(f"The directory {full_path.parent} does not exist.")

        # Compile only: a PDF save does not need the PNG render step.
        self._compile_pdf()
        shutil.copy2(self._pdf_path, full_path)

    def save_png(self, filepath: str | Path) -> None:
        """Saves the rendered LaTeX document as a PNG file.

        Args:
            filepath (str | Path): The file path to save the PNG to.

        Returns:
            None

        Raises:
            TypeError: If ``filepath`` is not a string or Path.
            ValueError: If ``filepath`` does not end in ``.png``.
            FileNotFoundError: If the destination directory does not exist.
            RuntimeError: If LaTeX compilation or PDF-to-PNG rendering fails.
        """
        if not isinstance(filepath, (str, Path)):
            raise TypeError("Path must be a string or Path object.")

        if not str(filepath).endswith(".png"):
            raise ValueError("File extension must be '.png'")

        full_path = Path(filepath).resolve()

        if not full_path.parent.exists():
            raise FileNotFoundError(f"The directory {full_path.parent} does not exist.")

        self._render_to_temp_png()
        shutil.copy2(self._png_path, full_path)
