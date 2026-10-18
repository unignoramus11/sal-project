import unicodedata

from sal import phones as P

W2V2 = "ɔːɹtᵻtʃɔːɹtɪsəplænslʌkɪweɪɡæftɪkaɪdeɪɹi"
ZIPA = "ɪspʰoːʈoːmẽsaːmn̪eːlɪkʰaɛhɛːpibəɾɖeːgʊbaːɾeːl̪əgeː"


def test_lossless():
    for s in (W2V2, ZIPA, "ojaːpakutʃl̪aʈikkeːt̪ikaːjʈeːɾekt̪aɾapu", "▁ɪs pʰoː"):
        assert "".join(P.segment(s)) == P.normalise(s)


def test_diacritics_attach():
    segs = P.segment(ZIPA)
    assert "pʰ" in segs and "oː" in segs and "kʰ" in segs and "n̪" in segs and "l̪" in segs
    assert unicodedata.normalize("NFD", "ẽ") in segs
    assert "ɡ" in segs and "g" not in "".join(segs)


def test_affricates_and_unknown_letters_kept():
    segs = P.segment(W2V2)
    assert "tʃ" in segs and "ᵻ" in segs
    assert P.segment("dʒa") == ["dʒ", "a"]


def test_dpr_counts_both_notations_once():
    # ZIPA style doubling and wav2vec2 style length mark give the same answer
    assert P.dpr("kulla") == (1, 2)        # k, ll -> 2 consonants, 1 geminate
    assert P.dpr("kulːa") == (1, 2)
    assert P.dpr("kullla") == (1, 2)       # a run of three is still one geminate
    assert P.dpr("aaa") == (0, 0)
    assert P.dpr("") == (0, 0)


def test_geminate_aspirate():
    # Bangru [ɾək.kʰaː]: closure k + aspirated release kʰ is one geminate aspirate
    assert P.segment("ɾəkkʰaː") == ["ɾ", "ə", "k", "kʰ", "aː"]
    assert P.dpr("ɾəkkʰaː") == (1, 2)      # ɾ, kkʰ
    assert P.dpr("ɾəkʰaː") == (0, 2)       # standard Hindi [ɾə.kʰaː]: no geminate
    assert P.dpr("pʰoːʈoː") == (0, 2)
    assert P.dpr("kʰkʰa") == (1, 1)


def test_different_consonants_are_not_a_run():
    assert P.dpr("akta") == (0, 2)
    assert P.dpr("aktʃa") == (0, 2)


def test_orphan_modifier_is_not_a_consonant():
    from sal.phones import dpr
    assert dpr("ːʈoː") == (0, 1)        # length mark of a vowel cut off by a window
    assert dpr("ʰkʰa") == (0, 1)
    assert dpr("oʈʈoː") == (1, 1)
