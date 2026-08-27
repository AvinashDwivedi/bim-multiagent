import pytest

from bim_agents.graph.client import _lucene_query


def test_lucene_search_text_removes_reserved_syntax_without_losing_unicode():
    assert _lucene_query('"מפסק הגנה" + panel:(A/B)') == "מפסק הגנה panel A B"


def test_lucene_search_text_rejects_only_reserved_characters():
    with pytest.raises(ValueError):
        _lucene_query('"+():')
