"""src/llm/weather.py — 기상청(KMA) API 클라이언트 (requests 모킹, 예외 전파 없음)."""
from unittest.mock import MagicMock, patch

from llm import weather


def _ncst_response():
    return {
        "response": {
            "header": {"resultCode": "00", "resultMsg": "NORMAL_SERVICE"},
            "body": {"items": {"item": [
                {"category": "T1H", "obsrValue": "23.5"},
                {"category": "REH", "obsrValue": "60"},
                {"category": "RN1", "obsrValue": "0"},
            ]}},
        }
    }


def _fcst_response():
    return {
        "response": {
            "header": {"resultCode": "00", "resultMsg": "NORMAL_SERVICE"},
            "body": {"items": {"item": [
                {"category": "TMN", "fcstDate": "20260703", "fcstTime": "0600", "fcstValue": "15"},
                {"category": "TMX", "fcstDate": "20260703", "fcstTime": "1500", "fcstValue": "27"},
                {"category": "TMP", "fcstDate": "20260703", "fcstTime": "0600", "fcstValue": "16"},
                {"category": "REH", "fcstDate": "20260703", "fcstTime": "0600", "fcstValue": "70"},
                {"category": "POP", "fcstDate": "20260703", "fcstTime": "0600", "fcstValue": "20"},
                {"category": "SKY", "fcstDate": "20260703", "fcstTime": "0600", "fcstValue": "1"},
            ]}},
        }
    }


def _error_response(code="03"):
    return {"response": {"header": {"resultCode": code, "resultMsg": "SERVICE KEY IS NOT REGISTERED"},
                          "body": {}}}


def setup_function(_):
    weather._CACHE.clear()


def test_to_grid_seoul():
    assert weather._to_grid(37.5665, 126.9780) == (60, 127)


def test_get_current_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("KMA_SERVICE_KEY", raising=False)
    r = weather.get_current()
    assert r["unavailable"] is True


def test_get_forecast_3d_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("KMA_SERVICE_KEY", raising=False)
    r = weather.get_forecast_3d()
    assert r["unavailable"] is True


def test_get_current_parses_response(monkeypatch):
    monkeypatch.setenv("KMA_SERVICE_KEY", "dummy-key")
    with patch("requests.get", return_value=MagicMock(
            status_code=200, json=lambda: _ncst_response())) as m:
        r = weather.get_current(37.5665, 126.9780)
    assert r["unavailable"] is False
    assert r["temp"] == 23.5
    assert r["humidity"] == 60.0
    assert r["rain"] == 0.0
    m.assert_called_once()


def test_get_forecast_3d_parses_response(monkeypatch):
    monkeypatch.setenv("KMA_SERVICE_KEY", "dummy-key")
    with patch("requests.get", return_value=MagicMock(
            status_code=200, json=lambda: _fcst_response())):
        r = weather.get_forecast_3d(37.5665, 126.9780)
    assert r["unavailable"] is False
    assert r["daily"][0]["tmn"] == 15.0
    assert r["daily"][0]["tmx"] == 27.0
    assert r["hourly"][0]["temp"] == 16.0
    assert r["hourly"][0]["humidity"] == 70.0
    assert r["hourly"][0]["pop"] == 20.0
    assert r["hourly"][0]["sky"] == "1"


def test_result_code_error_is_unavailable(monkeypatch):
    monkeypatch.setenv("KMA_SERVICE_KEY", "dummy-key")
    with patch("requests.get", return_value=MagicMock(
            status_code=200, json=lambda: _error_response())):
        r = weather.get_current(37.5665, 126.9780)
    assert r["unavailable"] is True


def test_http_exception_is_graceful(monkeypatch):
    monkeypatch.setenv("KMA_SERVICE_KEY", "dummy-key")
    with patch("requests.get", side_effect=Exception("network down")):
        r = weather.get_current(37.5665, 126.9780)
    assert r["unavailable"] is True


def test_cache_avoids_second_call(monkeypatch):
    monkeypatch.setenv("KMA_SERVICE_KEY", "dummy-key")
    with patch("requests.get", return_value=MagicMock(
            status_code=200, json=lambda: _ncst_response())) as m:
        weather.get_current(37.5665, 126.9780)
        weather.get_current(37.5665, 126.9780)
    assert m.call_count == 1


def test_forecast_cache_avoids_second_call(monkeypatch):
    monkeypatch.setenv("KMA_SERVICE_KEY", "dummy-key")
    with patch("requests.get", return_value=MagicMock(
            status_code=200, json=lambda: _fcst_response())) as m:
        weather.get_forecast_3d(37.5665, 126.9780)
        weather.get_forecast_3d(37.5665, 126.9780)
    assert m.call_count == 1
