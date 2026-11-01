from fastapi import Request


async def get_service(request: Request):
    return request.app.state.service
