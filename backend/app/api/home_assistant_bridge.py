"""Only the paired HACS integration can serve the outbound bridge."""
from contextlib import suppress
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect


def create_bridge_router(bridge):
    router = APIRouter()

    @router.post("/home-assistant/bridge/register", tags=["home-assistant"])
    async def register(request: Request):
        authorization = request.headers.get("Authorization", "")
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Falta la clave del vínculo.")
        payload = await request.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("bridge_id"), str):
            raise HTTPException(status_code=400, detail="La identificación del vínculo no es válida.")
        try:
            bridge.register(payload.get("bridge_id", ""), authorization[7:])
        except PermissionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return {"registered": True}

    @router.websocket("/home-assistant/bridge")
    async def socket(websocket: WebSocket):
        authorization = websocket.headers.get("Authorization", "")
        if not authorization.startswith("Bearer ") or not bridge.authorized(authorization[7:]):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        connection = await bridge.attach(websocket)
        try:
            while connection.connected:
                connection.receive(await websocket.receive_json())
        except (WebSocketDisconnect, ValueError):
            pass
        finally:
            bridge.detach(connection)
            with suppress(RuntimeError, WebSocketDisconnect):
                await websocket.close(code=1001)

    return router
