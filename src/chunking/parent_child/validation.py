def validate_configuration(
    parent_size: int,
    child_size: int,
    child_overlap: int,
) -> None:
    """Validate token budgets shared by parent-child implementations."""

    if parent_size <= 0:
        raise ValueError("parent_size must be greater than 0")
    if child_size <= 0:
        raise ValueError("child_size must be greater than 0")
    if parent_size <= child_size:
        raise ValueError("parent_size must be greater than child_size")
    if child_overlap < 0:
        raise ValueError("child_overlap cannot be negative")
    if child_overlap >= child_size:
        raise ValueError("child_overlap must be smaller than child_size")
