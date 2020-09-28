from logging import ERROR

from ssnapshot.ssnapshot import (
    create_job_summaries,
    create_job_detail_summary,
    create_partition_summary,
    dhhmmss_to_seconds,
    expand_compressed_slurm_nodelist,
    get_sinfo,
    get_squeue,
    get_sstat,
    seconds_to_hhmmss,
)


class TestCreateJobSummaries:
    def test_create_job_summaries(self):
        assert False


class TestCreateJobDetailSummary:
    pass


class TestCreatePartitionSummary:
    pass


class TestFunction_dhhmmss_to_seconds:
    def test_bad_input_returns_zero(self):
        assert dhhmmss_to_seconds('a') == 0
        assert dhhmmss_to_seconds('a-2') == 0
        assert dhhmmss_to_seconds('-1') == 0
        assert dhhmmss_to_seconds('') == 0
        assert dhhmmss_to_seconds([]) == 0
        assert dhhmmss_to_seconds(1) == 0
        assert dhhmmss_to_seconds([':', ':']) == 0

    def test_bad_input_logs_error(self, caplog):
        dhhmmss_to_seconds('a')
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelno == ERROR
        assert record.msg.find('Error trying to convert') != -1

    def test_seconds_only_returns_correct(self):
        assert dhhmmss_to_seconds('0') == 0
        assert dhhmmss_to_seconds('1') == 1
        assert dhhmmss_to_seconds('100') == 100

    def test_minutes_seconds_returns_correct(self):
        assert dhhmmss_to_seconds('1:00') == 60
        assert dhhmmss_to_seconds('10:10') == 610
        assert dhhmmss_to_seconds('111:00') == 6660

    def test_hours_minutes_seconds_returns_correct(self):
        assert dhhmmss_to_seconds('1:00:00') == 1*60*60
        assert dhhmmss_to_seconds('01:01:01') == 1*60*60 + 60 + 1

    def test_days_minutes_hours_seconds_returns_correct(self):
        assert dhhmmss_to_seconds('1-00:00:00') == 24*60*60



class TestExpandCompressedSlurmNodeList:
    pass


class TestGetSinfo:
    pass


class TestGetSqueue:
    def test_get_squeue_returns_dataframe(self):
        assert False


class TestGetSstat:
    def test_get_sstat_returns_dataframe(self):
        assert False


class TestFunction_seconds_to_hhmmss:
    def test_seconds_gives_correct_output(self):
        assert seconds_to_hhmmss(45) == '0:00:45'

    def test_minutes_gives_correct_output(self):
        assert seconds_to_hhmmss(1*60+45) == '0:01:45'

    def test_hours_gives_correct_output(self):
        assert seconds_to_hhmmss(3*3600+2*60+1) == '3:02:01'
