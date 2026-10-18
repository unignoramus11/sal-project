from sal.gemination import left_context, target_positions, token_contexts


def test_contexts():
    assert target_positions("रखा") == [1] and left_context("रखा", 1) == "initial_schwa"
    assert left_context("दिखाई", 2) == "vowel"
    assert token_contexts(["सकते", "उसके", "दुकान"], ["त", "क", "क"]) == ["medial_schwa", "medial_schwa", "vowel"]
    assert token_contexts(["रखा"], ["क"]) == ["unknown"]
