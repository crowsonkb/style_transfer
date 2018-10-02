import asyncio
import binascii
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path

from aiohttp import web
from aiohttp_index import IndexMiddleware
import numpy as np
from PIL import Image


MODULE_DIR = Path(__file__).parent.resolve()
STATIC_PATH = MODULE_DIR / 'web_static'


@dataclass
class Iterate:
    """A message containing a new iterate."""
    step: int
    steps: int
    time: float
    update_size: float
    loss: float
    tv: float
    image: Image.Image


@dataclass
class IterationFinished:
    """A message to notify the client that iteration has stopped."""


# pylint: disable=redefined-builtin
def pil_to_data_url(image, format='png', **kwargs):
    mime_types = {'jpeg': 'image/jpeg', 'png': 'image/png'}
    header = f'data:{mime_types[format]};base64,'
    buf = io.BytesIO()
    image.save(buf, format=format, **kwargs)
    return header + binascii.b2a_base64(buf.getvalue()).decode()


# pylint: disable=method-hidden
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
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
        if app.wss:
            msg = asdict(event)
            msg['_type'] = type(event).__name__
            if 'image' in msg:
                msg['image'] = pil_to_data_url(msg['image'], **app.image_encode_settings)
            await send_message(app, msg)


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
        self.app.image_encode_settings = {'format': 'png'}
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
