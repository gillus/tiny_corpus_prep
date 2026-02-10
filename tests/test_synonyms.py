"""Tests for tiny_corpus_prep.synonyms."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from tiny_corpus_prep.synonyms import SynonymMapper


class TestSynonymMapper:
    def test_init_empty(self):
        mapper = SynonymMapper()
        assert mapper.simplify_line("hello world") == "hello world"

    def test_init_with_mapping(self):
        mapper = SynonymMapper({"utilize": "use"})
        assert mapper.simplify_line("utilize this") == "use this"

    def test_case_preservation_upper(self):
        mapper = SynonymMapper({"utilize": "use"}, preserve_case=True)
        assert mapper.simplify_line("UTILIZE this") == "USE this"

    def test_case_preservation_title(self):
        mapper = SynonymMapper({"utilize": "use"}, preserve_case=True)
        assert mapper.simplify_line("Utilize this") == "Use this"

    def test_no_case_preservation(self):
        mapper = SynonymMapper({"utilize": "use"}, preserve_case=False)
        assert mapper.simplify_line("Utilize this") == "use this"

    def test_whole_word_matching(self):
        """Should not replace substrings."""
        mapper = SynonymMapper({"cat": "dog"})
        result = mapper.simplify_line("category concatenate cat")
        # "cat" should be replaced, but "category"/"concatenate" should not
        assert "dog" in result
        assert "category" in result
        assert "concatenate" in result

    def test_from_json(self, tmp_path: Path):
        mapping = {"old": "new", "big": "large"}
        json_path = tmp_path / "map.json"
        json_path.write_text(json.dumps(mapping))
        mapper = SynonymMapper.from_json(str(json_path))
        assert mapper.simplify_line("the old big house") == "the new large house"

    def test_add(self):
        mapper = SynonymMapper()
        mapper.add("foo", "bar")
        assert mapper.simplify_line("foo baz") == "bar baz"

    def test_simplify_iter(self):
        mapper = SynonymMapper({"hello": "hi"})
        result = list(mapper.simplify_iter(["hello world", "hello there"]))
        assert result == ["hi world", "hi there"]
