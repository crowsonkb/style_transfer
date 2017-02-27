from collections import namedtuple
import queue
import shlex
import threading

from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

Display = namedtuple('Display', 'type')
Exit = namedtuple('Exit', '')
Skip = namedtuple('Skip', '')


class Prompt:
    def __init__(self):
        self.history = InMemoryHistory()
        self.q = queue.Queue()
        self.shutdown = threading.Event()
        self.thread = threading.Thread(target=self.run)

    def start(self):
        self.thread.start()

    def stop(self):
        self.shutdown.set()

    def __del__(self):
        self.stop()

    def run(self):
        while not self.shutdown.is_set():
            try:
                text = prompt('> ', history=self.history, patch_stdout=True, refresh_interval=1/60)
                cmd = shlex.split(text)
                if not cmd:
                    continue
                elif cmd[0] in ('exit', 'quit'):
                    self.q.put(Exit())
                    return
                elif cmd[0] == 'help':
                    print('Help text forthcoming.')
                elif cmd[0] == 'skip':
                    self.q.put(Skip())
                else:
                    print('Unknown command. Try \'help\'.')
            except KeyboardInterrupt:
                continue
            except EOFError:
                self.q.put(Exit())
                return


class PromptResponder:
    def __init__(self, q):
        self.q = q

    def __call__(self):
        try:
            while True:
                event = self.q.get(block=False)
                if isinstance(event, Exit):
                    raise KeyboardInterrupt()
                elif isinstance(event, Skip):
                    return event
        except queue.Empty:
            pass
