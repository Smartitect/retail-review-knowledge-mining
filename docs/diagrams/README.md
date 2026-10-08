# Diagrams

The diagrams in [`architecture.md`](../architecture.md) and [`data-model.md`](../data-model.md), in endjin's visual style, each in a light and a dark variant. GitHub shows whichever matches the reader's theme.

| Diagram | Used in | Light | Dark |
|---|---|---|---|
| The pipeline and its caches | `architecture.md`, The pipeline | [`pipeline.svg`](pipeline.svg) | [`pipeline-dark.svg`](pipeline-dark.svg) |
| Package dependencies | `architecture.md`, Packages | [`packages.svg`](packages.svg) | [`packages-dark.svg`](packages-dark.svg) |
| How a batch is committed | `architecture.md`, How a batch is committed | [`batch-commit.svg`](batch-commit.svg) | [`batch-commit-dark.svg`](batch-commit-dark.svg) |
| The retail data model | `data-model.md` | [`data-model.svg`](data-model.svg) | [`data-model-dark.svg`](data-model-dark.svg) |

## Changing a diagram

The `.html` file is the source; the `.svg` beside it is exported from it.

1. Edit the `.html` file, and make the same change in its `-dark.html` twin. The two share one layout and differ only in colours.
2. Open it in a browser to check it.
3. Export the SVG. With the diagram-design Claude Code skill installed, run its `export_svg.py` script on the file, or ask your coding agent to export it. Without the skill, copy the `<svg>` element out of the HTML into the `.svg` file.
4. Check both variants on GitHub, in light and dark themes.

Keep to the style the diagrams already use: positions on a 4px grid, straight or right-angled connectors only, labels on a background mask beside their line rather than on it, and endjin's colours, with green reserved for the one or two things each diagram is about.

GitHub shows SVG images in a sandbox that cannot load web fonts, so on GitHub the text falls back from Inter to the system sans-serif. The colours and layout are exact.
