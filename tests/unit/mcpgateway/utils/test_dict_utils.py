# Third-Party
import pytest

# First-Party
from mcpgateway.utils.dict_utils import flatten_dict, flatten_schema, unflatten_dict


def test_flatten_unflatten_dict():
    nested = {"user": {"name": "Alice", "address": {"city": "Wonderland", "zip": "12345"}}, "id": 1}

    flat = flatten_dict(nested)
    assert flat == {"user.name": "Alice", "user.address.city": "Wonderland", "user.address.zip": "12345", "id": 1}

    unflat = unflatten_dict(flat)
    assert unflat == nested


def test_flatten_schema():
    schema = {
        "type": "object",
        "properties": {
            "user": {"type": "object", "properties": {"name": {"type": "string"}, "details": {"type": "object", "properties": {"bio": {"type": "string"}}, "required": ["bio"]}}, "required": ["name"]},
            "id": {"type": "integer"},
        },
        "required": ["user", "id"],
    }

    flat_schema = flatten_schema(schema)
    assert "user.name" in flat_schema["properties"]
    assert "user.details.bio" in flat_schema["properties"]
    assert "id" in flat_schema["properties"]

    # Check if required fields are correctly flattened
    assert "user.name" in flat_schema["required"]
    assert "user.details.bio" in flat_schema["required"]
    assert "id" in flat_schema["required"]

    # The actual schema should have no nested 'properties'
    for k, v in flat_schema["properties"].items():
        assert "properties" not in v
