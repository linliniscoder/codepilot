from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "benchmark_40.json"


@dataclass(frozen=True)
class TaskVariant:
    task_id: str
    issue: str
    hidden_test_filename: str
    hidden_test_source: str
    hidden_test_command: str


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    repo: str
    public_test_command: str
    hidden_test_source: str
    expected_files: tuple[str, ...]
    forbidden_files: tuple[str, ...]
    variants: tuple[TaskVariant, ...]


def render_module(import_block: str, tests: list[tuple[str, str]]) -> str:
    lines = [import_block.strip(), ""]
    for test_name, assertion in tests:
        lines.append(f"def test_{test_name}() -> None:")
        lines.append(f"    {assertion}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def math_average_suite() -> SuiteSpec:
    hidden_dir = "evaluation/hidden_tests/math_average"
    return SuiteSpec(
        name="math_average",
        repo="evaluation/fixtures/math_average",
        public_test_command="python -m pytest tests -q",
        hidden_test_source=hidden_dir,
        expected_files=("calculator.py",),
        forbidden_files=("tests/test_calculator.py",),
        variants=(
            TaskVariant(
                "math-average-001",
                "修复 calculator.py：add 应该执行加法。",
                "case_01_add_positive.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_01_add_positive.py -q",
            ),
            TaskVariant(
                "math-average-002",
                "修复 calculator.py：add 应该正确处理负数。",
                "case_02_add_negative.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_02_add_negative.py -q",
            ),
            TaskVariant(
                "math-average-003",
                "修复 calculator.py：average([]) 应该返回 0。",
                "case_03_average_empty.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_03_average_empty.py -q",
            ),
            TaskVariant(
                "math-average-004",
                "修复 calculator.py：average 需要正确处理整数列表。",
                "case_04_average_integers.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_04_average_integers.py -q",
            ),
            TaskVariant(
                "math-average-005",
                "修复 calculator.py：average 需要正确处理浮点列表。",
                "case_05_average_floats.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_05_average_floats.py -q",
            ),
            TaskVariant(
                "math-average-006",
                "修复 calculator.py：average 需要正确处理单元素列表。",
                "case_06_average_singleton.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_06_average_singleton.py -q",
            ),
            TaskVariant(
                "math-average-007",
                "修复 calculator.py：average 需要正确处理正负混合数据。",
                "case_07_average_mixed.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_07_average_mixed.py -q",
            ),
            TaskVariant(
                "math-average-008",
                "修复 calculator.py：add 和 average 都需要符合预期。",
                "case_08_average_large.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_08_average_large.py -q",
            ),
        ),
    )


def text_slugify_suite() -> SuiteSpec:
    hidden_dir = "evaluation/hidden_tests/text_slugify"
    return SuiteSpec(
        name="text_slugify",
        repo="evaluation/fixtures/text_slugify",
        public_test_command="python -m pytest tests -q",
        hidden_test_source=hidden_dir,
        expected_files=("text_utils.py",),
        forbidden_files=("tests/test_text_utils.py",),
        variants=(
            TaskVariant(
                "text-slugify-001",
                "修复 text_utils.py：slugify 需要处理基本英文句子。",
                "case_01_sentence.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_01_sentence.py -q",
            ),
            TaskVariant(
                "text-slugify-002",
                "修复 text_utils.py：slugify 需要去除首尾短横线。",
                "case_02_trim.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_02_trim.py -q",
            ),
            TaskVariant(
                "text-slugify-003",
                "修复 text_utils.py：连续分隔符应折叠为一个短横线。",
                "case_03_repeated_separators.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_03_repeated_separators.py -q",
            ),
            TaskVariant(
                "text-slugify-004",
                "修复 text_utils.py：标点输入应该返回空字符串。",
                "case_04_empty.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_04_empty.py -q",
            ),
            TaskVariant(
                "text-slugify-005",
                "修复 text_utils.py：纯标点输入应该返回空字符串。",
                "case_05_punctuation_only.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_05_punctuation_only.py -q",
            ),
            TaskVariant(
                "text-slugify-006",
                "修复 text_utils.py：空格和标点混合时应正确生成 slug。",
                "case_06_spaces_punctuation.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_06_spaces_punctuation.py -q",
            ),
            TaskVariant(
                "text-slugify-007",
                "修复 text_utils.py：数字和字母混合时应保留数字。",
                "case_07_numbers.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_07_numbers.py -q",
            ),
            TaskVariant(
                "text-slugify-008",
                "修复 text_utils.py：首尾空白和标点需要同时处理。",
                "case_08_trim_and_punctuation.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_08_trim_and_punctuation.py -q",
            ),
        ),
    )


def order_total_suite() -> SuiteSpec:
    hidden_dir = "evaluation/hidden_tests/order_total"
    return SuiteSpec(
        name="order_total",
        repo="evaluation/fixtures/order_total",
        public_test_command="python -m pytest tests -q",
        hidden_test_source=hidden_dir,
        expected_files=("order.py",),
        forbidden_files=("tests/test_order.py",),
        variants=(
            TaskVariant(
                "order-total-001",
                "修复 order.py：discount_percent 以百分数表示。",
                "case_01_percentage.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_01_percentage.py -q",
            ),
            TaskVariant(
                "order-total-002",
                "修复 order.py：0% 折扣应该保持原价。",
                "case_02_zero_discount.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_02_zero_discount.py -q",
            ),
            TaskVariant(
                "order-total-003",
                "修复 order.py：100% 折扣应该返回 0。",
                "case_03_full_discount.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_03_full_discount.py -q",
            ),
            TaskVariant(
                "order-total-004",
                "修复 order.py：需要正确处理小数金额。",
                "case_04_fractional_total.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_04_fractional_total.py -q",
            ),
            TaskVariant(
                "order-total-005",
                "修复 order.py：需要正确处理 12.5% 折扣。",
                "case_05_decimal_discount.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_05_decimal_discount.py -q",
            ),
            TaskVariant(
                "order-total-006",
                "修复 order.py：19.99 和 15% 的结果需要精确。",
                "case_06_precision.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_06_precision.py -q",
            ),
            TaskVariant(
                "order-total-007",
                "修复 order.py：小额金额也要保持精度。",
                "case_07_small_total.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_07_small_total.py -q",
            ),
            TaskVariant(
                "order-total-008",
                "修复 order.py：折扣值仍然按百分数计算。",
                "case_08_decimal_discount_precision.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_08_decimal_discount_precision.py -q",
            ),
        ),
    )


def missing_module_suite() -> SuiteSpec:
    hidden_dir = "evaluation/hidden_tests/missing_module"
    return SuiteSpec(
        name="missing_module",
        repo="evaluation/fixtures/missing_module",
        public_test_command="python -m pytest tests -q",
        hidden_test_source=hidden_dir,
        expected_files=("helpers.py",),
        forbidden_files=("tests/test_app.py",),
        variants=(
            TaskVariant(
                "missing-module-001",
                "修复 helpers.py：normalize_name 需要处理内层空白。",
                "case_01_trim.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_01_trim.py -q",
            ),
            TaskVariant(
                "missing-module-002",
                "修复 helpers.py：display_name 需要折叠多个空格。",
                "case_02_collapse_spaces.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_02_collapse_spaces.py -q",
            ),
            TaskVariant(
                "missing-module-003",
                "修复 helpers.py：制表符也需要被规范化。",
                "case_03_tabs.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_03_tabs.py -q",
            ),
            TaskVariant(
                "missing-module-004",
                "修复 helpers.py：换行和空白混合时应保持一个空格。",
                "case_04_newlines.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_04_newlines.py -q",
            ),
            TaskVariant(
                "missing-module-005",
                "修复 helpers.py：空字符串应保持为空字符串。",
                "case_05_empty.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_05_empty.py -q",
            ),
            TaskVariant(
                "missing-module-006",
                "修复 helpers.py：多余内部空白需要折叠。",
                "case_06_multiple_internal_spaces.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_06_multiple_internal_spaces.py -q",
            ),
            TaskVariant(
                "missing-module-007",
                "修复 helpers.py：混合空白输入要输出单个空格。",
                "case_07_mixed_whitespace.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_07_mixed_whitespace.py -q",
            ),
            TaskVariant(
                "missing-module-008",
                "修复 helpers.py：单个名字不应被修改。",
                "case_08_single_word.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_08_single_word.py -q",
            ),
        ),
    )


def word_count_suite() -> SuiteSpec:
    hidden_dir = "evaluation/hidden_tests/word_count"
    return SuiteSpec(
        name="word_count",
        repo="evaluation/fixtures/word_count",
        public_test_command="python -m pytest tests -q",
        hidden_test_source=hidden_dir,
        expected_files=("word_count.py",),
        forbidden_files=("tests/test_word_count.py",),
        variants=(
            TaskVariant(
                "word-count-001",
                "修复 word_count.py：count_words 需要处理单个空格。",
                "case_01_single_spaces.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_01_single_spaces.py -q",
            ),
            TaskVariant(
                "word-count-002",
                "修复 word_count.py：count_words 需要处理重复空格。",
                "case_02_repeated_spaces.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_02_repeated_spaces.py -q",
            ),
            TaskVariant(
                "word-count-003",
                "修复 word_count.py：空字符串应该返回 0。",
                "case_03_empty.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_03_empty.py -q",
            ),
            TaskVariant(
                "word-count-004",
                "修复 word_count.py：换行和制表符应该被正确计数。",
                "case_04_newlines_tabs.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_04_newlines_tabs.py -q",
            ),
            TaskVariant(
                "word-count-005",
                "修复 word_count.py：首尾空白不应影响词数。",
                "case_05_leading_trailing.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_05_leading_trailing.py -q",
            ),
            TaskVariant(
                "word-count-006",
                "修复 word_count.py：混合空白输入需要正确计数。",
                "case_06_mixed_whitespace.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_06_mixed_whitespace.py -q",
            ),
            TaskVariant(
                "word-count-007",
                "修复 word_count.py：单个单词应该返回 1。",
                "case_07_single_word.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_07_single_word.py -q",
            ),
            TaskVariant(
                "word-count-008",
                "修复 word_count.py：多个空行之间的词也要被统计。",
                "case_08_newline_separated.py",
                hidden_dir,
                "python -m pytest hidden_tests/case_08_newline_separated.py -q",
            ),
        ),
    )


def build_suite_specs() -> tuple[SuiteSpec, ...]:
    return (
        math_average_suite(),
        text_slugify_suite(),
        order_total_suite(),
        missing_module_suite(),
        word_count_suite(),
    )


def build_tasks(specs: tuple[SuiteSpec, ...]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for spec in specs:
        for variant in spec.variants:
            tasks.append(
                {
                    "id": variant.task_id,
                    "issue": variant.issue,
                    "repo": spec.repo,
                    "public_test_command": spec.public_test_command,
                    "hidden_test_command": variant.hidden_test_command,
                    "hidden_test_source": variant.hidden_test_source,
                    "expected_files": list(spec.expected_files),
                    "forbidden_files": list(spec.forbidden_files),
                }
            )
    return tasks


def build_hidden_test_content_map(
    specs: tuple[SuiteSpec, ...],
) -> dict[tuple[str, str], str]:
    content: dict[tuple[str, str], str] = {}

    math_import = "from calculator import add, average"
    math_cases = {
        "case_01_add_positive.py": render_module(
            math_import,
            [("add_positive_numbers", "assert add(2, 3) == 5")],
        ),
        "case_02_add_negative.py": render_module(
            math_import,
            [("add_negative_numbers", "assert add(-7, -4) == -11")],
        ),
        "case_03_average_empty.py": render_module(
            math_import,
            [("average_empty_values_returns_zero", "assert average([]) == 0")],
        ),
        "case_04_average_integers.py": render_module(
            math_import,
            [("average_integer_values", "assert average([2, 4, 6]) == 4")],
        ),
        "case_05_average_floats.py": render_module(
            math_import,
            [("average_float_values", "assert average([1.5, 2.5, 3.5]) == 2.5")],
        ),
        "case_06_average_singleton.py": render_module(
            math_import,
            [("average_single_value", "assert average([7]) == 7")],
        ),
        "case_07_average_mixed.py": render_module(
            math_import,
            [("average_mixed_signs", "assert average([-2, 4, -6, 8]) == 1")],
        ),
        "case_08_average_large.py": render_module(
            math_import,
            [("average_large_numbers", "assert average([1000000, 1000002]) == 1000001")],
        ),
    }

    slug_import = "from text_utils import slugify"
    slug_cases = {
        "case_01_sentence.py": render_module(
            slug_import,
            [("basic_sentence", 'assert slugify("Hello World") == "hello-world"')],
        ),
        "case_02_trim.py": render_module(
            slug_import,
            [("trim_outer_separators", 'assert slugify("  Python: Fast!  ") == "python-fast"')],
        ),
        "case_03_repeated_separators.py": render_module(
            slug_import,
            [("collapse_repeated_separators", 'assert slugify("one---two") == "one-two"')],
        ),
        "case_04_empty.py": render_module(
            slug_import,
            [("empty_text", 'assert slugify(" !!! ") == ""')],
        ),
        "case_05_punctuation_only.py": render_module(
            slug_import,
            [("punctuation_only", 'assert slugify("...") == ""')],
        ),
        "case_06_spaces_punctuation.py": render_module(
            slug_import,
            [("spaces_and_punctuation", 'assert slugify("Hello, world!!!") == "hello-world"')],
        ),
        "case_07_numbers.py": render_module(
            slug_import,
            [("numbers_are_preserved", 'assert slugify("v2.0 release") == "v2-0-release"')],
        ),
        "case_08_trim_and_punctuation.py": render_module(
            slug_import,
            [("trim_and_punctuation", 'assert slugify("  ---trim---  ") == "trim"')],
        ),
    }

    order_import = "from decimal import Decimal\n\nfrom order import total_after_discount"
    order_cases = {
        "case_01_percentage.py": render_module(
            order_import,
            [("percentage_discount", 'assert total_after_discount(Decimal("100"), Decimal("20")) == Decimal("80")')],
        ),
        "case_02_zero_discount.py": render_module(
            order_import,
            [("zero_discount", 'assert total_after_discount(Decimal("45"), Decimal("0")) == Decimal("45")')],
        ),
        "case_03_full_discount.py": render_module(
            order_import,
            [("full_discount", 'assert total_after_discount(Decimal("80"), Decimal("100")) == Decimal("0")')],
        ),
        "case_04_fractional_total.py": render_module(
            order_import,
            [("fractional_total", 'assert total_after_discount(Decimal("19.99"), Decimal("15")) == Decimal("16.9915")')],
        ),
        "case_05_decimal_discount.py": render_module(
            order_import,
            [("decimal_discount", 'assert total_after_discount(Decimal("200"), Decimal("12.5")) == Decimal("175.0")')],
        ),
        "case_06_precision.py": render_module(
            order_import,
            [("precision_is_preserved", 'assert total_after_discount(Decimal("19.99"), Decimal("5")) == Decimal("18.9905")')],
        ),
        "case_07_small_total.py": render_module(
            order_import,
            [("small_total", 'assert total_after_discount(Decimal("0.99"), Decimal("10")) == Decimal("0.891")')],
        ),
        "case_08_decimal_discount_precision.py": render_module(
            order_import,
            [("decimal_discount_precision", 'assert total_after_discount(Decimal("10"), Decimal("7.5")) == Decimal("9.25")')],
        ),
    }

    missing_import = "from app import display_name"
    missing_cases = {
        "case_01_trim.py": render_module(
            missing_import,
            [("normalize_inner_spaces", 'assert display_name("Ada   Lovelace") == "Ada Lovelace"')],
        ),
        "case_02_collapse_spaces.py": render_module(
            missing_import,
            [("collapse_multiple_spaces", 'assert display_name("Grace    Hopper") == "Grace Hopper"')],
        ),
        "case_03_tabs.py": render_module(
            missing_import,
            [("collapse_tabs", 'assert display_name("\\tAda\\tLovelace\\t") == "Ada Lovelace"')],
        ),
        "case_04_newlines.py": render_module(
            missing_import,
            [("collapse_newlines", 'assert display_name("  Alan\\nTuring  ") == "Alan Turing"')],
        ),
        "case_05_empty.py": render_module(
            missing_import,
            [("blank_input", 'assert display_name("   ") == ""')],
        ),
        "case_06_multiple_internal_spaces.py": render_module(
            missing_import,
            [("multiple_internal_spaces", 'assert display_name("Jean    Luc   Picard") == "Jean Luc Picard"')],
        ),
        "case_07_mixed_whitespace.py": render_module(
            missing_import,
            [("mixed_whitespace", 'assert display_name("  Ada\\n\\tLovelace  ") == "Ada Lovelace"')],
        ),
        "case_08_single_word.py": render_module(
            missing_import,
            [("single_word", 'assert display_name("Ada") == "Ada"')],
        ),
    }

    word_import = "from word_count import count_words"
    word_cases = {
        "case_01_single_spaces.py": render_module(
            word_import,
            [("single_spaces", 'assert count_words("one two three") == 3')],
        ),
        "case_02_repeated_spaces.py": render_module(
            word_import,
            [("repeated_spaces", 'assert count_words("one   two") == 2')],
        ),
        "case_03_empty.py": render_module(
            word_import,
            [("empty_string", 'assert count_words("") == 0')],
        ),
        "case_04_newlines_tabs.py": render_module(
            word_import,
            [("newlines_and_tabs", 'assert count_words("one\\ttwo\\nthree") == 3')],
        ),
        "case_05_leading_trailing.py": render_module(
            word_import,
            [("leading_and_trailing_whitespace", 'assert count_words("  one two  ") == 2')],
        ),
        "case_06_mixed_whitespace.py": render_module(
            word_import,
            [("mixed_whitespace", 'assert count_words("one \\n two \\t three") == 3')],
        ),
        "case_07_single_word.py": render_module(
            word_import,
            [("single_word", 'assert count_words("word") == 1')],
        ),
        "case_08_newline_separated.py": render_module(
            word_import,
            [("newline_separated_words", 'assert count_words("one\\n\\n two") == 2')],
        ),
    }

    content.update({("math_average", name): text for name, text in math_cases.items()})
    content.update({("text_slugify", name): text for name, text in slug_cases.items()})
    content.update({("order_total", name): text for name, text in order_cases.items()})
    content.update({("missing_module", name): text for name, text in missing_cases.items()})
    content.update({("word_count", name): text for name, text in word_cases.items()})

    expected_keys = {
        (spec.name, variant.hidden_test_filename)
        for spec in specs
        for variant in spec.variants
    }
    actual_keys = set(content)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            "hidden test content map mismatch: "
            f"missing={missing} extra={extra}"
        )
    return content


def write_hidden_tests(content_map: dict[tuple[str, str], str]) -> None:
    for (suite, filename), content in content_map.items():
        target = PROJECT_ROOT / "evaluation" / "hidden_tests" / suite / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def build_output_payload(specs: tuple[SuiteSpec, ...]) -> list[dict[str, Any]]:
    return build_tasks(specs)


def main() -> int:
    specs = build_suite_specs()
    hidden_tests = build_hidden_test_content_map(specs)
    write_hidden_tests(hidden_tests)
    tasks = build_output_payload(specs)
    OUTPUT_PATH.write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(tasks)} tasks to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
