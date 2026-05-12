"""Tests pour l'interface Property + PropertyContext + Registry."""

from __future__ import annotations

import pytest
import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import PropertyRegistry


# -----------------------------------------------------------------------------
# PropertyContext
# -----------------------------------------------------------------------------


def test_context_cache_is_lazy() -> None:
    ctx = PropertyContext()
    n_calls = 0

    def expensive() -> int:
        nonlocal n_calls
        n_calls += 1
        return 42

    assert ctx.get_or_compute("key", expensive) == 42
    assert ctx.get_or_compute("key", expensive) == 42  # cached
    assert n_calls == 1


def test_context_cache_key_deterministic() -> None:
    ctx = PropertyContext()
    k1 = ctx.cache_key("svd", (4, 8, 32, 32), "torch.float64")
    k2 = ctx.cache_key("svd", (4, 8, 32, 32), "torch.float64")
    assert k1 == k2
    k3 = ctx.cache_key("svd", (4, 8, 64, 64), "torch.float64")
    assert k1 != k3


def test_context_clear_cache() -> None:
    ctx = PropertyContext()
    ctx.cache["a"] = 1
    ctx.clear_cache()
    assert ctx.cache == {}


# -----------------------------------------------------------------------------
# Property interface conformity
# -----------------------------------------------------------------------------


def test_property_subclass_must_define_name() -> None:
    with pytest.raises(TypeError, match="name"):
        class BadProp(Property):
            family = "A"
            cost_class = 1

            def compute(self, A, ctx):
                return {}


def test_property_subclass_must_define_valid_family() -> None:
    with pytest.raises(TypeError, match="family"):
        class BadFamily(Property):
            name = "x"
            family = "Z9"  # pas une lettre A-W
            cost_class = 1

            def compute(self, A, ctx):
                return {}


def test_property_subclass_must_define_valid_cost_class() -> None:
    with pytest.raises(TypeError, match="cost_class"):
        class BadCost(Property):
            name = "x"
            family = "A"
            cost_class = 99

            def compute(self, A, ctx):
                return {}


def test_property_subclass_metadata_ok() -> None:
    class GoodProp(Property):
        name = "test_prop"
        family = "A"
        cost_class = 2

        def compute(self, A, ctx):
            return {"value": 1.0}

    p = GoodProp()
    assert p.name == "test_prop"
    assert p.family == "A"
    assert p.compute(None, PropertyContext()) == {"value": 1.0}


# -----------------------------------------------------------------------------
# PropertyRegistry
# -----------------------------------------------------------------------------


def test_registry_basic() -> None:
    reg = PropertyRegistry()

    class P1(Property):
        name = "p1"
        family = "A"
        cost_class = 1
        def compute(self, A, ctx): return {}

    class P2(Property):
        name = "p2"
        family = "B"
        cost_class = 3
        def compute(self, A, ctx): return {}

    reg.register(P1)
    reg.register(P2)

    assert len(reg) == 2
    assert reg.get("p1") is P1
    assert reg.get("p2") is P2


def test_registry_filter_by_family() -> None:
    reg = PropertyRegistry()

    class P1(Property):
        name = "p1"; family = "A"; cost_class = 1
        def compute(self, A, ctx): return {}

    class P2(Property):
        name = "p2"; family = "B"; cost_class = 3
        def compute(self, A, ctx): return {}

    reg.register(P1)
    reg.register(P2)

    a_only = reg.filter(family="A")
    assert a_only == [P1]


def test_registry_filter_by_cost_class_max() -> None:
    reg = PropertyRegistry()

    class P1(Property):
        name = "p1"; family = "A"; cost_class = 1
        def compute(self, A, ctx): return {}

    class P2(Property):
        name = "p2"; family = "B"; cost_class = 4
        def compute(self, A, ctx): return {}

    reg.register(P1)
    reg.register(P2)

    cheap = reg.filter(cost_class_max=2)
    assert cheap == [P1]


def test_registry_rejects_duplicate_name() -> None:
    reg = PropertyRegistry()

    class P1(Property):
        name = "dup"; family = "A"; cost_class = 1
        def compute(self, A, ctx): return {}

    class P2(Property):
        name = "dup"; family = "B"; cost_class = 3
        def compute(self, A, ctx): return {}

    reg.register(P1)
    with pytest.raises(ValueError, match="déjà enregistrée"):
        reg.register(P2)


def test_registry_unknown_raises_keyerror() -> None:
    reg = PropertyRegistry()
    with pytest.raises(KeyError, match="inconnue"):
        reg.get("nonexistent")
