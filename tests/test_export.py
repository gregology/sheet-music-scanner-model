from sheet_model.export import core_ml, tokens
from sheet_model.lines import HEIGHT, WIDTH
from sheet_model.model import STEPS, Reader

import numpy as np


# A reader with a layer or a loop Core ML cannot convert could not ship, however well it reads: the model must convert
# as it is (AGENTS.md). A small reader of the same design converts in seconds.
def test_the_reader_converts_to_core_ml_with_its_shapes():
    reader = core_ml(Reader(12, width=32, layers=1, heads=2))
    description = reader.get_spec().description
    assert [i.name for i in description.input] == ["picture"]
    assert list(description.input[0].type.multiArrayType.shape) == [1, 1, HEIGHT, WIDTH]
    assert [o.name for o in description.output] == ["scores"]


# Reading the Core ML reader's scores by a rule other than the PyTorch reader's would count lines as different that
# are the same, or the same that differ.
def test_the_check_reads_scores_by_ctcs_rule():
    best = np.array([[0, 3, 3, 0, 3, 5, 5, 0] + [0] * (STEPS - 8)])
    assert tokens(best) == [[3, 3, 5]]
