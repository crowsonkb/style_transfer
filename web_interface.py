import asyncio
import binascii
import io
import json
from pathlib import Path

import aiohttp
from aiohttp import web
from aiohttp_index import IndexMiddleware
import numpy as np
from PIL import Image

MODULE_DIR = Path(__file__).parent.resolve()
STATIC_PATH = MODULE_DIR / 'web_static'


def pil_to_data_url(image):
    header = 'data:image/png;base64,'
    buf = io.BytesIO()
    image.save(buf, format='png')
    return header + binascii.b2a_base64(buf.getvalue()).decode()


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, Image.Image):
            return pil_to_data_url(o)
        return super().default(o)


json_encoder = JSONEncoder()


async def handle_websocket(request):
    app = request.app
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    app.wss.append(ws)

    async for _ in ws:
        pass

    try:
        app.wss.remove(ws)
    except ValueError:
        pass

    return ws


async def send_message(app, msg):
    for ws in app.wss:
        try:
            await ws.send_json(msg, dumps=json_encoder.encode)
        except ConnectionError:
            try:
                app.wss.remove(ws)
            except ValueError:
                pass


async def process_events(app):
    while True:
        event = await app.event_queue.get()
        await send_message(app, event._asdict())


class WebInterface:
    def __init__(self):
        self.app = None
        self.loop = None

    def run(self, port=8000):
        """Runs the web interface."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.app = web.Application(middlewares=[IndexMiddleware()], loop=self.loop)
        self.app.event_queue = asyncio.Queue()
        self.app.wss = []
        self.app.task_process_events = self.loop.create_task(process_events(self.app))

        self.app.router.add_route('GET', '/websocket', handle_websocket)
        self.app.router.add_static('/', STATIC_PATH)

        try:
            web.run_app(self.app, port=port, shutdown_timeout=1, handle_signals=False)
        except KeyboardInterrupt:
            pass

    def put_event(self, event):
        self.loop.call_soon_threadsafe(self.app.event_queue.put_nowait, event)
