"""Tests for ArxivRetriever."""

import time
from types import SimpleNamespace

import feedparser
import pytest
import requests

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


@pytest.fixture()
def mock_arxiv_batches(monkeypatch):
    """Model independent API batches without network requests or real waits."""
    calls, waits, warnings = [], [], []

    def setup(outcomes, paper_count=60):
        feed = feedparser.FeedParserDict(
            feed=feedparser.FeedParserDict(title="arXiv"),
            entries=[feedparser.FeedParserDict(id=f"oai:arXiv.org:{i}") for i in range(paper_count)],
        )
        monkeypatch.setattr(arxiv_retriever.feedparser, "parse", lambda _: feed)

        class FakeClient:
            def __init__(self, **kwargs):
                assert kwargs["num_retries"] == 0

            def results(self, search):
                start = int(search.id_list[0])
                calls.append(start)
                sequence = outcomes.get(start, [])
                outcome = sequence.pop(0) if sequence else None
                if isinstance(outcome, Exception):
                    raise outcome
                if callable(outcome):
                    return outcome()
                return iter(search.id_list)

        monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)
        monkeypatch.setattr(arxiv_retriever, "sleep", waits.append)
        monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
        return calls, waits, warnings

    return setup


def _api_error(status):
    return arxiv_retriever.arxiv.HTTPError("https://export.arxiv.org/api/query", 0, status)


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_arxiv_retries_transient_http_errors(config, mock_arxiv_batches, status):
    calls, waits, _ = mock_arxiv_batches({0: [_api_error(status)]}, paper_count=20)
    assert ArxivRetriever(config)._retrieve_raw_papers() == [str(i) for i in range(20)]
    assert calls == [0, 0]
    assert waits == [30]


@pytest.mark.parametrize("error", [requests.exceptions.Timeout(), requests.exceptions.ConnectionError()])
def test_arxiv_retries_network_errors(config, mock_arxiv_batches, error):
    calls, waits, _ = mock_arxiv_batches({0: [error]}, paper_count=20)
    assert len(ArxivRetriever(config)._retrieve_raw_papers()) == 20
    assert calls == [0, 0]
    assert waits == [30]


def test_arxiv_recovers_when_rate_limit_changes_to_503(config, mock_arxiv_batches):
    calls, waits, _ = mock_arxiv_batches({0: [_api_error(429), _api_error(503)]}, paper_count=20)
    assert len(ArxivRetriever(config)._retrieve_raw_papers()) == 20
    assert calls == [0, 0, 0]
    assert waits == [30, 60]


def test_arxiv_retries_unexpected_empty_api_pages(config, mock_arxiv_batches):
    error = arxiv_retriever.arxiv.UnexpectedEmptyPageError("https://export.arxiv.org/api/query", 0, {})
    calls, waits, _ = mock_arxiv_batches({0: [error]}, paper_count=20)
    assert len(ArxivRetriever(config)._retrieve_raw_papers()) == 20
    assert calls == [0, 0]
    assert waits == [30]


def test_arxiv_preserves_successful_batches_and_continues_after_failure(config, mock_arxiv_batches):
    calls, waits, warnings = mock_arxiv_batches(
        {20: [_api_error(429)] + [_api_error(503) for _ in range(4)]}
    )
    papers = ArxivRetriever(config)._retrieve_raw_papers()
    assert papers == [str(i) for i in list(range(20)) + list(range(40, 60))]
    assert calls == [0] + [20] * 5 + [40]
    assert [wait for wait in waits if wait != 3] == [30, 60, 120, 120]
    assert any("incomplete" in warning for warning in warnings)


def test_arxiv_total_outage_raises_and_stops_after_three_batches(config, mock_arxiv_batches):
    calls, _, _ = mock_arxiv_batches(
        {start: [_api_error(503) for _ in range(5)] for start in range(0, 80, 20)},
        paper_count=80,
    )
    with pytest.raises(arxiv_retriever.arxiv.HTTPError) as exc:
        ArxivRetriever(config)._retrieve_raw_papers()
    assert exc.value.status == 503
    assert calls == [0] * 5 + [20] * 5 + [40] * 5


def test_arxiv_keeps_earlier_results_when_outage_persists(config, mock_arxiv_batches):
    calls, _, warnings = mock_arxiv_batches(
        {start: [_api_error(503) for _ in range(5)] for start in [20, 40, 60]},
        paper_count=100,
    )
    assert ArxivRetriever(config)._retrieve_raw_papers() == [str(i) for i in range(20)]
    assert 80 not in calls
    assert any("Stopping" in warning for warning in warnings)


def test_arxiv_does_not_hide_permanent_http_errors(config, mock_arxiv_batches):
    calls, _, _ = mock_arxiv_batches({20: [_api_error(403)]})
    with pytest.raises(arxiv_retriever.arxiv.HTTPError) as exc:
        ArxivRetriever(config)._retrieve_raw_papers()
    assert exc.value.status == 403
    assert calls == [0, 20]


def test_arxiv_genuine_empty_feed_returns_no_papers(config, mock_arxiv_batches):
    calls, _, _ = mock_arxiv_batches({}, paper_count=0)
    assert ArxivRetriever(config)._retrieve_raw_papers() == []
    assert calls == []


def test_arxiv_partial_iterator_retry_does_not_duplicate_papers(config, mock_arxiv_batches):
    def interrupted_batch():
        yield "0"
        raise _api_error(503)

    mock_arxiv_batches({0: [interrupted_batch]}, paper_count=20)
    assert ArxivRetriever(config)._retrieve_raw_papers() == [str(i) for i in range(20)]


def test_partial_arxiv_failure_still_reaches_email(config, mock_arxiv_batches, monkeypatch):
    """Exercise real retrieval, ranking, rendering and sending with offline services."""
    import smtplib
    from email import message_from_string, policy
    from zotero_arxiv_daily.executor import Executor
    from tests.canned_responses import (
        make_sample_paper, make_stub_openai_client, make_stub_smtp, make_stub_zotero_client,
    )

    mock_arxiv_batches({20: [_api_error(429)] + [_api_error(503) for _ in range(4)]})
    monkeypatch.setattr(
        ArxivRetriever, "convert_to_paper",
        lambda self, raw: make_sample_paper(title=f"Recovered paper {raw}", score=None),
    )
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: make_stub_zotero_client())
    client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: client)
    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))

    Executor(config).run()

    assert len(sent) == 1
    body = message_from_string(sent[0][2], policy=policy.default).get_content()
    assert "Recovered paper 0" in body
    assert "Recovered paper 40" in body
    assert "Recovered paper 20" not in body


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    # The RSS fixture gives us paper IDs.  After feedparser, the code calls
    # arxiv.Client().results(search) which makes real HTTP requests.  We mock
    # the arxiv Client so the test stays offline.
    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]
    paper_ids = [e.id.removeprefix("oai:arXiv.org:") for e in new_entries]

    # Build fake ArxivResult-like objects matching each RSS entry
    fake_results = []
    for entry in new_entries:
        pid = entry.id.removeprefix("oai:arXiv.org:")
        fake_results.append(SimpleNamespace(
            title=entry.title,
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url=f"https://arxiv.org/pdf/{pid}",
            entry_id=f"https://arxiv.org/abs/{pid}",
            source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
        ))

    class FakeClient:
        def __init__(self, **kw):
            pass
        def results(self, search):
            return iter(fake_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    # Skip file downloads in convert_to_paper
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=10, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=10, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]
