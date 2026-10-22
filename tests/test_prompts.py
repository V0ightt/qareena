from qareena.config import Settings, load_runtime
from qareena.readers.prompts import LOCATE, crop_prompt, field_schema, json_keys, user_prompt

RT = load_runtime(Settings(_env_file=None))


def test_every_document_has_unique_printed_labels_as_json_keys():
    for doc in RT.catalog.documents.values():
        keys = json_keys(doc)
        assert sorted(keys.values()) == sorted(doc.fields)
        assert list(field_schema(doc)["properties"]) == list(keys)


def test_prompt_explains_bilingual_labels_and_never_lists_english_keys():
    doc = RT.catalog.documents["iq_national_card_back"]
    text = user_prompt(doc)
    assert LOCATE in text
    assert "issuing_authority" not in text


def test_crop_prompt_names_the_label_and_its_kurdish_translation():
    doc = RT.catalog.documents["iq_national_card_back"]
    text = crop_prompt(doc, "issuing_authority")
    assert doc.fields["issuing_authority"].label_ar in text
    assert doc.fields["issuing_authority"].aliases[0] in text
