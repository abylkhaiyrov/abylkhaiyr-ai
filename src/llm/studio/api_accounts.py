from typing import Annotated, Any

from fastapi import APIRouter, Cookie, Depends, HTTPException, Query, Request, Response, status
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from llm.studio.accounts import AuthenticationError, public_user
from llm.studio.protocol import API_VERSION, API_VERSION_HEADER
from llm.studio.security import (
    CSRF_COOKIE,
    SESSION_COOKIE,
    Dependencies,
    Identity,
    SlidingWindowLimiter,
    client_address,
)
from llm.studio.service import StudioService

SESSION_MAX_AGE = 12 * 60 * 60
STATUS_BY_CODE = {
    "invalid_credentials": status.HTTP_401_UNAUTHORIZED,
    "account_disabled": status.HTTP_403_FORBIDDEN,
    "email_taken": status.HTTP_409_CONFLICT,
    "username_taken": status.HTTP_409_CONFLICT,
    "account_conflict": status.HTTP_409_CONFLICT,
    "last_admin": status.HTTP_409_CONFLICT,
}


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class RegisterRequest(CamelModel):
    first_name: str = Field(min_length=1, max_length=64)
    last_name: str = Field(min_length=1, max_length=64)
    email: str = Field(min_length=3, max_length=254)
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=512)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    identifier: str = Field(
        min_length=1, max_length=254, validation_alias=AliasChoices("identifier", "username")
    )
    password: str = Field(min_length=1, max_length=512)


class PasswordChangeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current_password: str = Field(min_length=1, max_length=512)
    new_password: str = Field(min_length=1, max_length=512)


class ProfileRequest(CamelModel):
    first_name: str = Field(min_length=1, max_length=64)
    last_name: str = Field(min_length=1, max_length=64)
    email: str = Field(min_length=3, max_length=254)
    username: str = Field(min_length=1, max_length=64)


class UserUpdateRequest(CamelModel):
    first_name: str | None = Field(default=None, max_length=64)
    last_name: str | None = Field(default=None, max_length=64)
    email: str | None = Field(default=None, max_length=254)
    username: str | None = Field(default=None, max_length=64)
    role: str | None = Field(default=None, max_length=16)
    is_active: bool | None = None


def fail(error: AuthenticationError) -> HTTPException:
    return HTTPException(
        status_code=STATUS_BY_CODE.get(error.code, status.HTTP_400_BAD_REQUEST), detail=error.code
    )


def rate_limited() -> HTTPException:
    return HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="rate_limited")


def set_session_cookies(response: Response, token: str, csrf: str) -> None:
    for name, value, http_only in ((SESSION_COOKIE, token, True), (CSRF_COOKIE, csrf, False)):
        response.set_cookie(
            name,
            value,
            httponly=http_only,
            secure=False,
            samesite="strict",
            max_age=SESSION_MAX_AGE,
            path="/",
        )


def clear_session_cookies(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")
    response.delete_cookie(CSRF_COOKIE, path="/")


def create_account_router(studio: StudioService, deps: Dependencies) -> APIRouter:
    router = APIRouter()
    login_by_address = SlidingWindowLimiter(10, 300)
    login_by_account = SlidingWindowLimiter(5, 300)
    register_by_address = SlidingWindowLimiter(5, 3600)
    register_total = SlidingWindowLimiter(60, 3600)
    Session = Annotated[Identity, Depends(deps.session)]
    Csrf = Annotated[Identity, Depends(deps.csrf)]
    AdminRead = Annotated[Identity, Depends(deps.admin)]
    AdminWrite = Annotated[Identity, Depends(deps.admin_csrf)]

    @router.post("/auth/register", status_code=status.HTTP_201_CREATED)
    def register(payload: RegisterRequest, request: Request, response: Response) -> dict[str, Any]:
        address = client_address(request)
        if not register_by_address.allow(address) or not register_total.allow("all"):
            raise rate_limited()
        register_by_address.hit(address)
        register_total.hit("all")
        try:
            token, csrf, user = studio.auth.register(
                payload.first_name,
                payload.last_name,
                payload.email,
                payload.username,
                payload.password,
                request.headers.get("user-agent"),
            )
        except AuthenticationError as error:
            raise fail(error) from error
        set_session_cookies(response, token, csrf)
        return public_user(user)

    @router.post("/auth/login")
    def login(
        payload: LoginRequest,
        request: Request,
        response: Response,
        session_token: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
    ) -> dict[str, Any]:
        address, account = client_address(request), payload.identifier.strip().lower()
        if not login_by_address.allow(address) or not login_by_account.allow(account):
            raise rate_limited()
        try:
            token, csrf, user = studio.auth.authenticate(
                payload.identifier,
                payload.password,
                request.headers.get("user-agent"),
                session_token,
            )
        except AuthenticationError as error:
            login_by_address.hit(address)
            login_by_account.hit(account)
            raise fail(error) from error
        login_by_address.clear(address)
        login_by_account.clear(account)
        set_session_cookies(response, token, csrf)
        return public_user(user)

    @router.get("/auth/me")
    def me(identity: Session, response: Response) -> dict[str, Any]:
        response.headers[API_VERSION_HEADER] = API_VERSION
        response.headers["Cache-Control"] = "no-store"
        return public_user(identity[0])

    @router.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
    def logout(
        _: Csrf,
        response: Response,
        session_token: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
    ) -> Response:
        studio.auth.logout(session_token)
        clear_session_cookies(response)
        response.status_code = status.HTTP_204_NO_CONTENT
        return response

    @router.put("/auth/password", status_code=status.HTTP_204_NO_CONTENT)
    def change_password(
        payload: PasswordChangeRequest, identity: Csrf, response: Response
    ) -> Response:
        try:
            studio.auth.change_password(
                identity[0].id, payload.current_password, payload.new_password
            )
        except AuthenticationError as error:
            raise fail(error) from error
        clear_session_cookies(response)
        response.status_code = status.HTTP_204_NO_CONTENT
        return response

    @router.patch("/auth/profile")
    def update_profile(payload: ProfileRequest, identity: Csrf) -> dict[str, Any]:
        try:
            user = studio.auth.update_profile(
                identity[0].id,
                payload.first_name,
                payload.last_name,
                payload.email,
                payload.username,
            )
        except AuthenticationError as error:
            raise fail(error) from error
        return public_user(user)

    @router.get("/auth/sessions")
    def sessions(
        identity: Session, session_token: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None
    ) -> list[dict[str, Any]]:
        return studio.auth.list_sessions(identity[0].id, session_token)

    @router.post("/auth/sessions/revoke-others")
    def revoke_other_sessions(
        identity: Csrf, session_token: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None
    ) -> dict[str, int]:
        return {"revoked": studio.auth.revoke_sessions(identity[0].id, session_token)}

    @router.delete("/auth/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
    def revoke_session(session_id: int, identity: Csrf) -> Response:
        if not studio.auth.revoke_session(identity[0].id, session_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session_not_found")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.get("/users")
    def list_users(
        _: AdminRead,
        query: str | None = Query(default=None, max_length=100),
        role: str | None = None,
        active: bool | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        return studio.users.search(query, role, active, limit, offset)

    @router.get("/users/{user_id}")
    def get_user(user_id: str, _: AdminRead) -> dict[str, Any]:
        user = studio.users.get(user_id)
        if user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user_not_found")
        return user

    @router.patch("/users/{user_id}")
    def update_user(user_id: str, payload: UserUpdateRequest, admin: AdminWrite) -> dict[str, Any]:
        changes = payload.model_dump(exclude_none=True)
        try:
            user = studio.users.update(admin[0].id, user_id, changes)
        except AuthenticationError as error:
            raise fail(error) from error
        if user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user_not_found")
        return user

    @router.post("/users/{user_id}/reset-password")
    def reset_password(user_id: str, admin: AdminWrite) -> dict[str, Any]:
        try:
            result = studio.users.reset_password(admin[0].id, user_id)
        except AuthenticationError as error:
            raise fail(error) from error
        if result is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user_not_found")
        return result

    @router.post("/users/{user_id}/revoke-sessions")
    def revoke_user_sessions(user_id: str, admin: AdminWrite) -> dict[str, int]:
        revoked = studio.users.revoke_sessions(admin[0].id, user_id)
        if revoked is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user_not_found")
        return {"revoked": revoked}

    @router.get("/notifications")
    def list_notifications(
        identity: Session,
        unread: bool = False,
        type: str | None = None,
        limit: int = 20,
        offset: int = 0,
        sort: str = "desc",
    ) -> dict[str, Any]:
        return studio.notifications.list(identity[0].id, unread, type, limit, offset, sort == "asc")

    @router.get("/notifications/unread-count")
    def unread_count(identity: Session) -> dict[str, int]:
        return {"unread": studio.notifications.unread_count(identity[0].id)}

    @router.post("/notifications/read-all")
    def read_all(identity: Csrf) -> dict[str, int]:
        return {"updated": studio.notifications.mark_all_read(identity[0].id)}

    @router.get("/notifications/{notification_id}")
    def get_notification(notification_id: str, identity: Session) -> dict[str, Any]:
        item = studio.notifications.get(identity[0].id, notification_id)
        if item is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="notification_not_found"
            )
        return item

    @router.post("/notifications/{notification_id}/read")
    def read_notification(notification_id: str, identity: Csrf) -> dict[str, Any]:
        item = studio.notifications.mark_read(identity[0].id, notification_id)
        if item is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="notification_not_found"
            )
        return item

    @router.delete("/notifications/{notification_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_notification(notification_id: str, identity: Csrf) -> Response:
        if not studio.notifications.delete(identity[0].id, notification_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="notification_not_found"
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router
