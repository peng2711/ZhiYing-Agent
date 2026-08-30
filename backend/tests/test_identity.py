import hashlib
import hmac

from starlette.requests import Request

from api.main import (
    _AUTH_SIGNATURE_HEADER,
    _AUTH_USER_HEADER,
    _authenticate_user_header,
    _resolve_chat_identity,
)


def make_request(headers=None, cookies=None):
    raw_headers = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    if cookies:
        raw_headers.append((b"cookie", "; ".join(f"{k}={v}" for k, v in cookies.items()).encode()))
    return Request({
        "type": "http",
        "method": "POST",
        "path": "/chat",
        "headers": raw_headers,
        "query_string": b"",
    })


def test_guest_identity_is_issued_and_reused(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    first = make_request()
    guest_id, is_guest, newly_issued = _resolve_chat_identity(first, "anonymous")
    assert is_guest is True
    assert newly_issued is True

    second = make_request(cookies={"zhiying_guest_id": guest_id})
    reused_id, reused_guest, new_cookie = _resolve_chat_identity(second, "anonymous")
    assert (reused_id, reused_guest, new_cookie) == (guest_id, True, False)


def test_signed_authenticated_user_is_mapped(monkeypatch):
    secret = "test-secret"
    user_id = "user_001"
    signature = hmac.new(secret.encode(), user_id.encode(), hashlib.sha256).hexdigest()
    monkeypatch.setenv("ZHIYING_USER_ID_SECRET", secret)

    request = make_request(headers={
        _AUTH_USER_HEADER: user_id,
        _AUTH_SIGNATURE_HEADER: signature,
    })
    assert _authenticate_user_header(request) is None
    assert request.state.authenticated_user_id == user_id

    resolved_id, is_guest, newly_issued = _resolve_chat_identity(request, "anonymous")
    assert (resolved_id, is_guest, newly_issued) == (user_id, False, False)


def test_invalid_authenticated_signature_is_rejected(monkeypatch):
    monkeypatch.setenv("ZHIYING_USER_ID_SECRET", "test-secret")
    request = make_request(headers={
        _AUTH_USER_HEADER: "user_001",
        _AUTH_SIGNATURE_HEADER: "bad",
    })
    error = _authenticate_user_header(request)
    assert error is not None
    assert error.status_code == 401
