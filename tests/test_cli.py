from safesense import cli
from safesense.cli import build_parser


def test_cli_exposes_planned_commands() -> None:
    parser = build_parser()
    args = parser.parse_args(["prepare", "--raw-dir", "data", "--cache-dir", "cache"])
    assert args.command == "prepare"
    train = parser.parse_args(["train", "--prepared-dir", "fusion", "--output-dir", "out"])
    assert train.command == "train"
    assert train.seeds == [11, 22, 33]
    final = parser.parse_args(["final", "--prepared-dir", "fusion", "--output-dir", "out", "--frozen-epoch", "4"])
    assert final.batch_size == 32
    assert parser.parse_args(["evaluate"]).command == "evaluate"
    assert parser.parse_args(["report"]).command == "report"
    fusion = parser.parse_args(["prepare-fusion", "--data-dir", "data", "--output-dir", "fusion"])
    assert fusion.command == "prepare-fusion"
    assert parser.parse_args(
        ["fusion-smoke", "--repo-dir", "upstream", "--prepared-dir", "fusion", "--output-dir", "out", "--seed", "22"]
    ).seed == 22
    assert parser.parse_args(
        ["fusion-reproduce", "--repo-dir", "upstream", "--prepared-dir", "fusion", "--output-dir", "out"]
    ).command == "fusion-reproduce"
    assert parser.parse_args(["smoke", "--prepared-dir", "fusion", "--output-dir", "out"]).command == "smoke"


def test_cli_propagates_failed_runtime_status(monkeypatch) -> None:
    monkeypatch.setattr(cli, "runtime_smoke", lambda *_args, **_kwargs: {"status": "failed"})
    monkeypatch.setattr(
        "sys.argv",
        ["safesense", "fusion-smoke", "--repo-dir", "r", "--prepared-dir", "p", "--output-dir", "o"],
    )
    assert cli.main() == 2
