from collector.eod import is_fyers_request_limit_error


def test_fyers_daily_limit_is_recognized_for_fallback():
    err = RuntimeError("fyers history failed: {'code': -360, 'message': \"You've reached the Standard plan limit of 5,000 requests/day\"}")
    assert is_fyers_request_limit_error(err)


def test_unrelated_provider_errors_do_not_trigger_quota_fallback():
    assert not is_fyers_request_limit_error(RuntimeError("connection reset"))
