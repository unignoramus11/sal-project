"""The rebuild must behave exactly like PRiSM's Vaani-Hi scripts where we claim it does."""
import io

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from sal.data import rebuild


def prism_create_splits(df):
    """Copy of PRiSM src/recipe/geolocation/local/vaanihindi_upload.py:create_splits.

    Verbatim except one line: the NaN column is made `object`, because pandas >= 3 refuses to write
    strings into a float column (older pandas, which PRiSM ran, upcast with a warning).
    """
    tr_frac = 0.75
    val_frac = 0.1
    np.random.seed(42)
    pincodes = df["pincode"].unique()
    df["split"] = np.nan
    df["split"] = df["split"].astype(object)  # the one change, see docstring
    for pincode in pincodes:
        pdf = df[df["pincode"] == pincode]
        n = len(pdf)
        pdf = pdf.sample(frac=1, random_state=42)  # shuffle
        n_train = int(n * tr_frac)
        n_val = int(n * val_frac)
        df.loc[pdf.index[:n_train], "split"] = "train"
        df.loc[pdf.index[n_train : n_train + n_val], "split"] = "val"
        df.loc[pdf.index[n_train + n_val :], "split"] = "test"
    df.loc[df["split"].isna(), "split"] = "train"
    return df


def prism_filter(df):
    """PRiSM vaanihindi_sample.py filtering loop (groups with > 450 rows, ascending pincode)."""
    filtered = [gdf for _, gdf in df.groupby("pincode") if gdf.shape[0] > 450]
    return pd.concat(filtered).reset_index(drop=True)


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    sizes = {110001: 900, 124001: 455, 124022: 451, 125001: 450, 800001: 1200, 302001: 30}
    rows = [{"pincode": p, "x": i} for p, n in sizes.items() for i in range(n)]
    df = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
    df["x"] = rng.permutation(len(df))
    return df


def test_filter_matches_prism(synthetic):
    ours = pd.concat([g for _, g in synthetic.groupby("pincode") if g.shape[0] > rebuild.UTT_PER_PINCODE],
                     ignore_index=True)
    ref = prism_filter(synthetic)
    pd.testing.assert_frame_equal(ours, ref)
    assert 125001 not in set(ours["pincode"])  # exactly 450 is dropped (strictly more than 450)
    assert list(ours["pincode"].drop_duplicates()) == sorted(ours["pincode"].unique())


def test_splits_match_prism(synthetic):
    df = prism_filter(synthetic)
    ours = rebuild.create_splits(df.copy())
    ref = prism_create_splits(df.copy())
    assert (ours["split"].values == ref["split"].values).all()


def test_split_fractions(synthetic):
    df = rebuild.create_splits(prism_filter(synthetic))
    for _, g in df.groupby("pincode"):
        n = len(g)
        c = g["split"].value_counts()
        assert c.get("train", 0) == int(n * 0.75)
        assert c.get("val", 0) == int(n * 0.1)
        assert c.get("test", 0) == n - int(n * 0.75) - int(n * 0.1)


def test_parse_coord():
    assert rebuild.parse_coord("28.6139") == pytest.approx(28.6139)
    assert rebuild.parse_coord("28.6N") == pytest.approx(28.6)
    assert rebuild.parse_coord("77.2E") == pytest.approx(77.2)
    assert rebuild.parse_coord("12.5S") == pytest.approx(-12.5)
    assert np.isnan(rebuild.parse_coord("NA"))


def _wav(seconds, sr=16000):
    buf = io.BytesIO()
    sf.write(buf, (np.sin(np.arange(int(seconds * sr)) / 7) * 8000).astype(np.int16), sr,
             subtype="PCM_16", format="WAV")
    return buf.getvalue()


def test_audio_kept_bit_identical():
    b = _wav(5.0)
    out, dur, changed = rebuild.prepare_audio(b)
    assert out == b and dur == pytest.approx(5.0) and not changed


def test_audio_clipped_to_20s():
    b = _wav(23.5)
    out, dur, changed = rebuild.prepare_audio(b)
    assert changed and dur == pytest.approx(20.0)
    x, sr = sf.read(io.BytesIO(out), dtype="int16")
    y, _ = sf.read(io.BytesIO(b), dtype="int16")
    assert sr == 16000 and (x == y[: 20 * sr]).all()


def test_speaker_from_path():
    p = "IISc_VaaniProject_M_Haryana_Rohtak_Aara10941c9eaf68_10941141917000099968_Rohtak-SPECIFIC_01962_6970_12520.wav"
    assert rebuild.speaker_from_path(p) == "Aara10941c9eaf68"
    with pytest.raises(ValueError):
        rebuild.speaker_from_path("something_else.wav")


def test_normalise_all_null_float_column():
    import pyarrow as pa
    t = pa.table({"speakerID": pa.array([None, None], pa.float64()), "language": ["Hindi", "Hindi"]})
    n = rebuild._normalise(t)
    assert pa.types.is_string(n.schema.field("speakerID").type) and n.column("speakerID").null_count == 2
    bad = pa.table({"speakerID": pa.array([1.0, None])})
    with pytest.raises(TypeError):
        rebuild._normalise(bad)


def test_robust_coordinates_ignore_misplaced_offices(tmp_path):
    rows = [("454552", 22.22, 75.47), ("454552", 22.19, 75.41), ("454552", 22.24, 75.42),   # one cluster
            ("454552", 27.22, 77.49), ("454552", 26.20, 83.46), ("454552", 46.16, 107.45),  # misplaced, out of box
            ("342310", 26.92, 70.91), ("342310", 26.75, 72.53), ("342310", 27.56, 71.25)]   # no cluster
    f = tmp_path / "pin.csv"
    pd.DataFrame(rows, columns=["Pincode", "Latitude", "Longitude"]).to_csv(f, index=False)
    r = rebuild.robust_pincode_coordinates(f)
    assert r.loc["454552", "cluster"] == 3 and r.loc["454552", "offices"] == 5
    assert abs(r.loc["454552", "latitude"] - 22.22) < 0.01 and abs(r.loc["454552", "longitude"] - 75.42) < 0.01
    assert r.loc["342310", "cluster"] == 3 and abs(r.loc["342310", "latitude"] - 26.92) < 0.01   # median of all
    mean = rebuild.pincode_coordinates(f)[0]["454552"]
    assert mean[0] > 23                                    # the mean is pulled 100+ km away
