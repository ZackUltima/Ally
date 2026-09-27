"""Object detector at a low rate → SceneObjects (FR-19).

Furniture boxes become suggested seat/bed zones, used only once the carer confirms them (ADR-011); objects
on the floor in the walkway region are flagged `on_floor`. Never escalates.

Sprint: S5. See docs/context/architecture.md and methods.md §6.4.
"""
