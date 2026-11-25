from sal.comparisons import standard


def test_transcript_pairs_need_both_configs():
    cfgs = {"tp_w2v2_rebuild_phone_gru_prism", "tp_w2v2_rebuild_phone_shuffle_prism", "tp_w2v2_rebuild_char_gru_prism"}
    got = {(a, b) for a, b, _ in standard(cfgs, "_prism")}
    assert got == {("tp_w2v2_rebuild_phone_gru_prism", "tp_w2v2_rebuild_phone_shuffle_prism"),
                   ("tp_w2v2_rebuild_phone_gru_prism", "tp_w2v2_rebuild_char_gru_prism")}


def test_suffix_keeps_splits_apart():
    cfgs = {"tp_zipa_rebuild_phone_gru_prism", "tp_zipa_rebuild_phone_shuffle_speaker_split"}
    assert standard(cfgs, "_prism") == [] and standard(cfgs, "_speaker_split") == []


def test_representation_pairs_per_layer_set():
    base = ["rp_w2v2_rebuild_L24_attn", "rp_w2v2_rebuild_L24_attn_pe", "rp_w2v2_rebuild_L24_gru",
            "rp_w2v2_rebuild_L24_chunkshuffle", "rp_w2v2_rebuild_L24_transformer", "rp_w2v2_rebuild_L1-2-4_gru",
            "rp_w2v2_rebuild_L1-2-4_chunkshuffle", "tp_w2v2_rebuild_char_gru", "tp_w2v2_rebuild_phone_gru"]
    got = {(a, b) for a, b, _ in standard(set(base))}
    assert ("rp_w2v2_rebuild_L24_gru", "rp_w2v2_rebuild_L24_chunkshuffle") in got
    assert ("rp_w2v2_rebuild_L24_gru", "rp_w2v2_rebuild_L24_attn") in got
    assert ("rp_w2v2_rebuild_L24_attn_pe", "rp_w2v2_rebuild_L24_attn") in got
    assert ("rp_w2v2_rebuild_L24_transformer", "rp_w2v2_rebuild_L24_attn") in got
    assert ("rp_w2v2_rebuild_L1-2-4_gru", "rp_w2v2_rebuild_L1-2-4_chunkshuffle") in got
    assert ("tp_w2v2_rebuild_char_gru", "rp_w2v2_rebuild_L24_attn") in got
    assert ("tp_w2v2_rebuild_phone_gru", "rp_w2v2_rebuild_L24_gru") in got
    # no cross-layer pairs, and attn_pe is not read as attn
    assert not any("L1-2-4" in a and "L24" in b for a, b in got)
    assert ("tp_w2v2_rebuild_phone_gru", "rp_w2v2_rebuild_L1-2-4_gru") in got
    assert len(got) == 9          # 7 above, the weighted-layer pair, and transcript phone vs char BiGRU


def test_representation_pairs_with_split_suffix_and_checks_excluded():
    cfgs = {"rp_zipa_rebuild_L7_gru_speaker_split", "rp_zipa_rebuild_L7_chunkshuffle_speaker_split",
            "rp_zipa_rebuild_L7_attn_prism_control", "rp_zipa_rebuild_L7_gru_prism_control"}
    got = standard(cfgs, "_speaker_split")
    assert [(a, b) for a, b, _ in got] == [("rp_zipa_rebuild_L7_gru_speaker_split",
                                            "rp_zipa_rebuild_L7_chunkshuffle_speaker_split")]
