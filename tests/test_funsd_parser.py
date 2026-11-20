import json

from cvdoc.datasets.funsd import assign_splits, parse_annotation


def test_funsd_parser(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(
        json.dumps(
            {
                "form": [
                    {"id": 1, "text": "Published text", "box": [1, 2, 50, 20]},
                    {"text": "", "box": [1, 2, 3, 4]},
                    {"text": "bad", "box": [1, 2, 1, 4]},
                ]
            }
        )
    )
    entities, counts = parse_annotation(path)
    assert entities[0]["text"] == "Published text"
    assert counts == {"empty_text": 1, "invalid_box": 1}


def test_funsd_split_preserved():
    train = [f"train{i}" for i in range(149)]
    test = [f"test{i}" for i in range(50)]
    splits = assign_splits(train, test)
    assert all(splits[i] == "test" for i in test)
    assert list(splits.values()).count("val") == 30
    assert list(splits.values()).count("dev") == 119
    assert splits == assign_splits(train[::-1], test[::-1])
