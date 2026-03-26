# Standard
import collections
from typing import Any, Dict, List, Optional, Union


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Recursively flatten a nested dictionary into a single-level dictionary with dot-notation keys.

    Args:
        d: The dictionary to flatten.
        parent_key: The base key to use for prefixing flattened keys.
        sep: The separator between nested keys (default is ".").

    Returns:
        A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, collections.abc.Mapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """
    Convert a flattened dictionary with dot-notation keys back into a nested dictionary.

    Args:
        d: The flattened dictionary.
        sep: The separator used in keys (default is ".").

    Returns:
        A nested dictionary.
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        target = result
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return result


def flatten_schema(schema: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten a nested JSON Schema's properties into a single-level properties dictionary.
    Exposes nested object fields as top-level dot-notation parameters.

    Args:
        schema: The JSON Schema to flatten.
        parent_key: The prefix for flattened property names.
        sep: The separator (default ".").

    Returns:
        A new JSON Schema with flattened properties and required fields.
    """
    if not isinstance(schema, dict):
        return schema

    # Often 'type' is missing but 'properties' is present in OpenAPI/Swagger
    has_properties = "properties" in schema
    is_object = schema.get("type") == "object" or has_properties

    if not is_object or not has_properties:
        return schema

    new_properties = {}
    new_required = []

    old_properties = schema.get("properties", {})
    old_required = schema.get("required", [])

    for k, v in old_properties.items():
        full_key = f"{parent_key}{sep}{k}" if parent_key else k

        # If the property is an object itself, recurse (flatten it)
        if isinstance(v, dict) and ("properties" in v or v.get("type") == "object"):
            flattened_sub = flatten_schema(v, full_key, sep=sep)
            sub_props = flattened_sub.get("properties", {})

            if sub_props:
                new_properties.update(sub_props)
                # Sub-required fields are already flattened in the sub-call
                new_required.extend(flattened_sub.get("required", []))

                # If the parent field itself was required, but we flattened it,
                # we should probably make its sub-fields required too?
                # Actually, if 'user' is required, then 'user.name' should be required IF 'name' was required in 'user'.
                # But what if 'user' was required but NONE of its sub-fields were?
                # In that case, we might want to make all of them required, or just leave it.
                # To be safe and avoid breaking changes, we only propagate ALREADY required sub-fields.
                if k in old_required and not flattened_sub.get("required"):
                    # If the parent was required but no sub-fields were marked required,
                    # we make all immediate sub-fields of this object required to satisfy the parent requirement.
                    new_required.extend(sub_props.keys())
            else:
                # Type object but no properties (e.g. additionalProperties: true)
                new_properties[full_key] = v
                if k in old_required:
                    new_required.append(full_key)
        else:
            new_properties[full_key] = v
            if k in old_required:
                new_required.append(full_key)

    # Clean duplicates
    new_required = sorted(list(set(new_required)))

    # Return a new schema object but keep top-level metadata
    res = schema.copy()
    res["type"] = "object"
    res["properties"] = new_properties
    res["required"] = new_required
    return res
