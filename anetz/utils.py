import difflib


def diff(a, b):
    """
    Displays changes in +/- changed lines format
    """
    diff = difflib.unified_diff(a.splitlines(1), b.splitlines(1))
    green = "\x1b[38;5;2m"
    red = "\x1b[38;5;1m"
    reset = "\x1b[0m"
    diff = [
        f"{green}{line}{reset}"
        if line.startswith("+")
        else f"{red}{line}{reset}"
        if line.startswith("-")
        else line
        for line in diff
    ]
    return "".join(diff)


def inline_diff(a: str, b: str) -> str:
    """
    Displays changes inline where possible (highlighted in green/red bg)
    rather than +/- changed lines format
    """
    output = []
    matcher = difflib.SequenceMatcher(None, a, b)
    green = "\x1b[38;5;16;48;5;2m"
    red = "\x1b[38;5;16;48;5;1m"
    reset = "\x1b[0m"

    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == "equal":
            output.append(a[a0:a1])
        elif opcode == "insert":
            output.append(f"{green}{b[b0:b1]}{reset}")
        elif opcode == "delete":
            output.append(f"{red}{a[a0:a1]}{reset}")
        elif opcode == "replace":
            output.append(f"{green}{b[b0:b1]}{reset}")
            output.append(f"{red}{a[a0:a1]}{reset}")
    return "".join(output)


def model_diff(auto_cls, model_a: str, model_b: str, diff_formatter=diff) -> str:
    """
    Displays changes in model structure between two models

    Example:

        from transformers import AutoModelForQuestionAnswering
        print(
            model_diff(
                AutoModelForQuestionAnswering, 'bert-base-uncased', 'roberta-base'
            )
        )
    """
    a = auto_cls.from_pretrained(model_a)
    b = auto_cls.from_pretrained(model_b)
    return diff_formatter(str(a), str(b))
