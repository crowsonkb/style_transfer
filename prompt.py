import ast
from collections import namedtuple
import queue
import shlex
import threading

from prompt_toolkit import CommandLineInterface
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import style_from_dict
from prompt_toolkit.shortcuts import create_eventloop, create_prompt_application
from prompt_toolkit.token import Token

Exit = namedtuple('Exit', '')
Set = namedtuple('Set', 'key value')
Skip = namedtuple('Skip', '')


class Prompt:
    def __init__(self, run_name, state_obj):
        self.run_name = run_name
        self.state_obj = state_obj
        self.cli = None
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self.run)

    def start(self):
        self.thread.start()

    def stop(self):
        if self.cli:
            self.cli.exit()
        self.thread.join()

    def get_bottom_toolbar_tokens(self, cli):
        return [(Token.Toolbar, 'Run '),
                (Token.Name, self.run_name),
                (Token.Toolbar, ' in progress.')]

    def get_prompt_tokens(self, cli):
        return [(Token.Prompt, '> ')]

    def run(self):
        style = style_from_dict({
            Token.Prompt: 'bold',
            Token.Toolbar: '#ccc bg:#333',
            Token.Name: '#fff bold bg:#333',
        })

        history = InMemoryHistory()
        eventloop = create_eventloop()
        app = create_prompt_application(history=history, style=style,
                                        get_bottom_toolbar_tokens=self.get_bottom_toolbar_tokens,
                                        get_prompt_tokens=self.get_prompt_tokens)
        self.cli = CommandLineInterface(app, eventloop)

        with self.cli.patch_stdout_context(raw=True):
            while True:
                try:
                    self.cli.run()
                    doc = self.cli.return_value()
                    if doc is None:
                        return
                    cmd = shlex.split(doc.text)
                    app.buffer.reset(append_to_history=True)

                    if not cmd:
                        continue
                    elif cmd[0] in ('exit', 'quit'):
                        self.q.put(Exit())
                        return
                    elif cmd[0] == 'help':
                        print('Help text forthcoming.')
                    elif cmd[0] == 'skip':
                        self.q.put(Skip())
                    elif cmd[0] == 'set':
                        self.q.put(Set(cmd[1], ast.literal_eval(' '.join(cmd[2:]))))
                    else:
                        print('Unknown command. Try \'help\'.')
                except KeyboardInterrupt:
                    continue
                except EOFError:
                    self.q.put(Exit())
                    return
                except Exception as err:
                    print(err)
                    self.q.put(Exit())
                    return


class PromptResponder:
    def __init__(self, q, args):
        self.q = q
        self.args = args

    def __call__(self):
        try:
            while True:
                event = self.q.get(block=False)
                if isinstance(event, Exit):
                    raise KeyboardInterrupt()
                elif isinstance(event, Skip):
                    return event
                elif isinstance(event, Set):
                    setattr(self.args, event.key, event.value)
        except queue.Empty:
            pass
