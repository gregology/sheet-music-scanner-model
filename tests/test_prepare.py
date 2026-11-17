from sheet_model.prepare import finished


# The generator once left only a log line for an example that failed its checks. Its shard held 999 examples, and the
# training that waits for finished shards waited for good. A shard still being filled must not count as finished.
def test_a_shard_with_a_failed_example_is_finished_and_one_still_filling_is_not(tmp_path):
    for number in range(999):
        (tmp_path / f"{number:08d}.json").write_text("{}")
    assert not finished(tmp_path)
    (tmp_path / "00000999.failed").write_text("RuntimeError: the render does not show the page its truth describes\n")
    assert finished(tmp_path)
