"""Build a tool catalog from inspect_ai tool signatures.

The catalog is a list of ToolTemplates — one per tool — derived
mechanically from parameter schemas. No model involvement. The catalog
is the action space that A* searches over.
"""

import logging

from inspect_ai.tool import Tool
from inspect_ai.tool._tool_info import parse_tool_info

from goap.types import ToolParam, ToolTemplate

logger = logging.getLogger(__name__)


def _tool_name(tool: Tool) -> str:
    from inspect_ai._util.registry import registry_info

    return registry_info(tool).name.split("/")[-1]


def build_catalog(tools: list[Tool]) -> list[ToolTemplate]:
    """Build a ToolTemplate catalog from inspect_ai tool signatures.

    Each tool gets:
    - params: extracted from parse_tool_info(), preserving name/type/required/description
    - output_type: ``{tool_name}_result`` (e.g. ``search_emails_result``)
    """
    catalog: list[ToolTemplate] = []

    for tool in tools:
        name = _tool_name(tool)
        info = parse_tool_info(tool)

        params: list[ToolParam] = []
        for pname, pschema in info.parameters.properties.items():
            ptype = "string"
            pdesc = getattr(pschema, "description", "") or ""

            if hasattr(pschema, "type") and pschema.type:
                ptype = pschema.type
                if ptype == "array" and hasattr(pschema, "items") and pschema.items:
                    item_type = getattr(pschema.items, "type", "string") or "string"
                    ptype = f"array of {item_type}"
            elif hasattr(pschema, "anyOf") and pschema.anyOf:
                # Union type (e.g. list[str] | None) — pick the non-null variant
                for variant in pschema.anyOf:
                    vtype = getattr(variant, "type", None)
                    if vtype and vtype != "null":
                        ptype = vtype
                        if vtype == "array" and hasattr(variant, "items") and variant.items:
                            item_type = getattr(variant.items, "type", "string") or "string"
                            ptype = f"array of {item_type}"
                        break
            elif isinstance(pschema, dict):
                ptype = pschema.get("type", "string")
                pdesc = pschema.get("description", "") or ""

            # Capture enum constraints
            penum = getattr(pschema, "enum", None)
            if penum:
                penum = tuple(penum)

            params.append(
                ToolParam(
                    name=pname,
                    type=ptype,
                    required=pname in info.parameters.required,
                    description=pdesc,
                    enum=penum,
                )
            )

        template = ToolTemplate(
            name=name,
            description=info.description,
            params=params,
            output_type=f"{name}_result",
        )
        catalog.append(template)
        logger.debug("Catalog: %s(%s) -> %s", name, ", ".join(p.name for p in params), template.output_type)

    logger.info("Built catalog with %d tool templates", len(catalog))
    return catalog


def format_catalog(catalog: list[ToolTemplate]) -> str:
    """Format catalog as compact notation for the trusted model prompt.

    Example output:
        search_emails(query: string, sender: string) -> search_emails_result
          Search for emails matching query and sender.
    """
    lines: list[str] = []
    for tmpl in catalog:
        def _fmt_param(p: ToolParam) -> str:
            s = f"{p.name}: {p.type}{'?' if not p.required else ''}"
            if p.enum:
                s += f" [{'/'.join(p.enum)}]"
            return s

        params_str = ", ".join(_fmt_param(p) for p in tmpl.params)
        lines.append(f"{tmpl.name}({params_str}) -> {tmpl.output_type}")
        if tmpl.description:
            # First line of description only
            desc_line = tmpl.description.strip().split("\n")[0]
            lines.append(f"  {desc_line}")
    return "\n".join(lines)
