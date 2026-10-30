import random
import re

import pytest

from litellm.proxy.spend_tracking.session_page import (
    Session,
    _parse_session_cursor,
    _Read,
    _Request,
    _session_page,
    read_session_page,
)

_Row = tuple[int, str, str]  # (startTime, session key, api_key)


def _ranked(rows: list[_Row], request: _Request) -> list[tuple[int, str, str]]:
    """The sessions in ``rows`` in page order, after the request's cursor or offset."""
    last_activity: dict[tuple[str, str], int] = {}
    for start, key, api_key in rows:
        last_activity[(key, api_key)] = max(last_activity.get((key, api_key), start), start)
    ranked = sorted(((at, key, api_key) for (key, api_key), at in last_activity.items()), reverse=request.sort_desc)
    if request.cursor is None:
        return ranked[request.offset :]
    cursor = (int(request.cursor[0]), request.cursor[1], request.cursor[2])
    return [session for session in ranked if (session < cursor if request.sort_desc else session > cursor)]


def _reader(rows: list[_Row], request: _Request, budgets: list[int | None], every_row: bool = False):
    """Reads like the page query: the newest ``budget`` rows, where a LIMIT cuts tied rows lowest key first, the
    opposite of page order, so a session at the read's oldest time can hide a tied session that ranks before it.
    With ``every_row`` it ignores the budget, which is what grouping every row returns."""
    newest_first = sorted(rows, key=lambda row: (-row[0], row[1], row[2]))

    async def read(budget: int | None) -> _Read:
        budgets.append(budget)
        taken = newest_first if budget is None or every_row else newest_first[:budget]
        page = _ranked(taken, request)[: request.page_limit + 1] if request.page_limit > 0 else []
        return _Read(
            candidates=tuple(Session(key, api_key, f"{at:06d}") for at, key, api_key in page),
            last_settled=bool(page) and page[-1][0] > taken[-1][0],
            rows_read=len(taken),
            session_count=len({(key, api_key) for _, key, api_key in taken}),
            read_everything=budget is None or every_row or len(taken) < budget,
        )

    return read


def _sessions(rng: random.Random) -> list[_Row]:
    """Sessions of one to forty rows over a short clock, so many rows share a time."""
    rows: list[_Row] = []
    clock = rng.choice((3, 40, 400))
    for index in range(rng.randrange(1, 120)):
        start, length = rng.randrange(0, clock), rng.choice((1, 1, 2, 3, 8, 40))
        rows.extend((start + rng.randrange(0, 30), f"s{index:03d}", rng.choice(("k1", "k2"))) for _ in range(length))
    return rows


def _requests(rng: random.Random, rows: list[_Row]):
    for _ in range(12):
        cap, page_size, sort_desc = rng.randrange(2, 60), rng.randrange(1, 12), rng.random() < 0.8
        mode = rng.choice(("first", "offset", "cursor"))
        full = _ranked(rows, _Request(None, 0, 0, sort_desc, cap))
        cursor = None
        if mode == "cursor" and full:
            at, key, api_key = full[rng.randrange(len(full))]
            cursor = (f"{at:06d}", key, api_key)
        offset = (rng.randrange(1, 2 * cap // page_size + 3) - 1) * page_size if mode == "offset" else 0
        yield _Request(cursor, offset, min(page_size, cap - offset), sort_desc, cap)


@pytest.mark.parametrize("seed", range(60))
@pytest.mark.asyncio
async def test_a_page_read_from_the_newest_rows_matches_grouping_every_row(seed):
    rng = random.Random(seed)
    rows = _sessions(rng)
    for request in _requests(rng, rows):
        grouped = await _session_page(_reader(rows, request, [], every_row=True), request)
        assert await _session_page(_reader(rows, request, []), request) == grouped, request


@pytest.mark.parametrize(
    "rows_per_session, expected_budgets",
    [
        pytest.param(1, [112], id="one-read-holds-the-cap"),
        pytest.param(2, [112, 896], id="grows-8x"),
        pytest.param(50, [112, 7168], id="skips-8x-when-its-density-cannot-pass-the-cap"),
        pytest.param(500, [112, None], id="skips-to-every-row"),
    ],
)
@pytest.mark.asyncio
async def test_reads_grow_8x_then_64x_then_every_row_skipping_reads_too_sparse_to_pass_the_cap(
    rows_per_session, expected_budgets
):
    """A cap of 100 and a page of 10 start at 100 + 10 + 2 rows."""
    rows = [(index, f"s{index // rows_per_session:05d}", "k") for index in range(150 * rows_per_session)]
    request = _Request(cursor=None, offset=0, page_limit=10, sort_desc=True, cap=100)
    budgets: list[int | None] = []
    await _session_page(_reader(rows, request, budgets), request)
    assert budgets == expected_budgets


@pytest.mark.asyncio
async def test_a_session_tied_with_the_reads_oldest_row_is_not_trusted():
    """Thirty one-row sessions share one time and the first read takes 17 of them. The page's sixth session sits at the
    read's oldest time, where the LIMIT may have cut a session that ranks before it, so the page reads on."""
    rows = [(0, f"s{index:02d}", "k") for index in range(30)]
    request = _Request(cursor=None, offset=0, page_limit=5, sort_desc=True, cap=10)
    budgets: list[int | None] = []
    page = await _session_page(_reader(rows, request, budgets), request)
    assert [session.key for session in page.sessions] == ["s29", "s28", "s27", "s26", "s25"]
    assert budgets == [17, 136]


@pytest.mark.parametrize("session_count, expected_total", [(100, (100, False)), (101, (100, True))])
@pytest.mark.asyncio
async def test_the_total_is_capped_only_past_the_cap(session_count, expected_total):
    rows = [(index, f"s{index:05d}", "k") for index in range(session_count)]
    request = _Request(cursor=None, offset=0, page_limit=10, sort_desc=True, cap=100)
    page = await _session_page(_reader(rows, request, []), request)
    assert (page.total, page.total_is_capped) == expected_total


@pytest.mark.asyncio
async def test_a_page_number_near_the_bigint_limit_reads_at_most_twice_the_cap():
    rows = [(index, f"s{index:05d}", "k") for index in range(500)]
    offset = (9_300_000_000_000_000 - 1) * 1000
    request = _Request(cursor=None, offset=offset, page_limit=min(1000, 100 - offset), sort_desc=True, cap=100)
    budgets: list[int | None] = []
    page = await _session_page(_reader(rows, request, budgets), request)
    assert budgets == [2 * 100 + 2]
    assert (page.sessions, page.total, page.total_is_capped) == ((), 100, True)


@pytest.mark.asyncio
async def test_an_ascending_page_reads_every_row_once():
    rows = [(index, f"s{index:05d}", "k") for index in range(500)]
    request = _Request(cursor=None, offset=0, page_limit=10, sort_desc=False, cap=100)
    budgets: list[int | None] = []
    await _session_page(_reader(rows, request, budgets), request)
    assert budgets == [None]


def test_parse_session_cursor():
    assert _parse_session_cursor(None) is None
    assert _parse_session_cursor("no-delimiter") is None
    assert _parse_session_cursor("2026-08-29 10:00:00|sess-1") is None
    assert _parse_session_cursor("|hashed-key|sess-1") is None
    assert _parse_session_cursor("2026-08-29 10:00:00|hashed-key|") is None
    assert _parse_session_cursor("2026-08-29 10:00:00|hashed-key|sess-1") == (
        "2026-08-29 10:00:00",
        "sess-1",
        "hashed-key",
    )
    assert _parse_session_cursor("2026-08-29 10:00:00.123|hashed-key|sess|with|pipes") == (
        "2026-08-29 10:00:00.123",
        "sess|with|pipes",
        "hashed-key",
    )


def _bound(sql: str, args: tuple[object, ...], pattern: str) -> tuple[object, ...]:
    """The values bound to the ``$N`` placeholders that ``pattern``'s groups capture in ``sql``."""
    found = re.search(pattern, sql)
    assert found, f"{pattern!r} not found in:\n{sql}"
    assert all(1 <= int(index) <= len(args) for index in found.groups()), f"{found.groups()} outside $1..${len(args)}"
    return tuple(args[int(index) - 1] for index in found.groups())


class _TableDb:
    """A table of ``rows`` matching rows in ``sessions`` sessions, whose reads return no page row."""

    def __init__(self, calls: list[tuple[str, tuple[object, ...]]], rows: int, sessions: int):
        self.calls, self.rows, self.sessions = calls, rows, sessions

    async def query_raw(self, query: str, *args: object):
        self.calls.append((query, args))
        budget = re.search(r"DESC LIMIT (\d+)", query)
        rows_read = min(int(budget.group(1)), self.rows) if budget else self.rows
        summary = {"rows_read": rows_read, "session_count": self.sessions}
        return [{"session_key": None, "api_key": None, "last_activity": None, "settled": None, **summary}]


@pytest.mark.asyncio
async def test_a_read_that_fills_its_budget_is_not_taken_for_every_row():
    """A read that took every row its budget allowed may have stopped short, so its 50 sessions are no total: the reads
    grow until one takes fewer rows than it asked for."""
    calls: list[tuple[str, tuple[object, ...]]] = []
    page = await read_session_page(
        _TableDb(calls, rows=9_000, sessions=50),
        "TRUE",
        (),
        "",
        page=1,
        page_size=10,
        sort_desc=True,
        session_cursor=None,
        cap=100,
    )
    budgets = [re.search(r"DESC LIMIT (\d+)", sql) for sql, _ in calls]
    assert [int(found.group(1)) if found else None for found in budgets] == [112, 896, 7168, None]
    assert (page.total, page.total_is_capped) == (50, False)


@pytest.mark.asyncio
async def test_a_page_starting_at_the_cap_reads_once_and_fetches_no_session():
    calls: list[tuple[str, tuple[object, ...]]] = []
    page = await read_session_page(
        _TableDb(calls, rows=1_000, sessions=150),
        "TRUE",
        (),
        "",
        page=5,
        page_size=25,
        sort_desc=True,
        session_cursor=None,
        cap=100,
    )
    assert len(calls) == 1, "the first read already holds more sessions than the cap"
    assert _bound(*calls[0], r"LIMIT \$(\d+)") == (0,)
    assert (page.sessions, page.has_more, page.total, page.total_is_capped) == ((), False, 100, True)


@pytest.mark.parametrize(
    "cap, expected_budgets",
    [
        pytest.param(103, [115, 920, 7360, None], id="13-sessions-per-115-rows-just-reach-104-in-920"),
        pytest.param(104, [116, 7424, None], id="13-sessions-per-116-rows-fall-one-short-of-105-in-928"),
    ],
)
@pytest.mark.asyncio
async def test_the_density_skip_reads_a_budget_whose_density_exactly_reaches_past_the_cap(cap, expected_budgets):
    """A budget is skipped only when its rows, as dense as the last read, would hold fewer than cap + 1 sessions."""
    calls: list[tuple[str, tuple[object, ...]]] = []
    await read_session_page(_TableDb(calls, rows=100_000, sessions=13), "TRUE", (), "", 1, 10, True, None, cap)
    budgets = [re.search(r"DESC LIMIT (\d+)", sql) for sql, _ in calls]
    assert [int(found.group(1)) if found else None for found in budgets] == expected_budgets


@pytest.mark.asyncio
async def test_the_read_binds_filters_cursor_limit_and_offset_to_their_placeholders():
    calls: list[tuple[str, tuple[object, ...]]] = []
    db = _TableDb(calls, rows=50, sessions=40)
    where, params = "model = $1 AND api_key = $2", ("gpt-4o", "key-a")
    await read_session_page(db, where, params, "", 1, 3, True, "2026-10-28 09:00:00|key-a|sess|9", cap=100)
    await read_session_page(db, where, params, "", 3, 1, True, None, cap=100)
    (cursor_sql, cursor_args), (offset_sql, offset_args) = calls
    for sql, args in calls:
        assert _bound(sql, args, r"model = \$(\d+) AND api_key = \$(\d+)") == params
        assert "page.last_activity > summary.oldest AS settled" in sql, (
            "a session tied with the read's oldest row may rank after a session the LIMIT cut, so it is not settled"
        )
    assert _bound(
        cursor_sql, cursor_args, r"\(last_activity, session_key, api_key\) < \(\$(\d+)::timestamp, \$(\d+), \$(\d+)\)"
    ) == ("2026-10-28 09:00:00", "sess|9", "key-a")
    assert _bound(cursor_sql, cursor_args, r"LIMIT \$(\d+)") == (4,)
    assert _bound(offset_sql, offset_args, r"LIMIT \$(\d+) OFFSET \$(\d+)") == (2, 2)
