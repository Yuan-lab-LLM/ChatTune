"""Task-scoped response compatibility for the packaged score filter (stdlib only)."""
from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import re
import secrets
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


SCORE_WITH_ECHO = re.compile(
    r'\A[ \t]*分数[ \t]*[:：][ \t]*(100|[1-9]?[0-9])[ \t]*'
    r'(?:(?:\r?\n|\\n)[ \t]*)+<DATA>.*\Z', re.DOTALL)


def normalize_score_response(content):
    """Accept only an explicit leading score followed immediately by echoed data."""
    if not isinstance(content, str):
        return content
    match = SCORE_WITH_ECHO.fullmatch(content)
    return f'分数：{match.group(1)}' if match else content


def is_scoring_request(body):
    messages = body.get('messages') if isinstance(body, dict) else None
    if not isinstance(messages, list) or len(messages) != 1 or body.get('stream'):
        return False
    message = messages[0]
    if not isinstance(message, dict) or message.get('role') != 'user':
        return False
    prompt = message.get('content')
    return (isinstance(prompt, str) and prompt.startswith(
        ('你是训练数据质量评分员。', '你是预训练语料（PT）数据质量评分员。'))
        and '分数：<0-100 的整数>' in prompt and '\n<DATA>\n' in prompt)


class ScoringProxy:
    def __init__(self, upstream, timeout):
        parsed = urlsplit(upstream)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.query or parsed.fragment:
            raise ValueError('Invalid scoring API base URL')
        self.upstream = upstream.rstrip('/')
        self.timeout = timeout
        self.prefix = '/' + secrets.token_hex(24)
        self.normalized_count = 0
        self.lock = threading.Lock()
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass  # Never log credentials, prompts, or model response contents.

            def reply(self, status, content, content_type='application/json', retry_after=None):
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(content)))
                if retry_after:
                    self.send_header('Retry-After', retry_after)
                self.end_headers()
                try:
                    self.wfile.write(content)
                except ConnectionError:
                    pass  # The scoring client may have timed out or been cancelled.

            def forward(self):
                route = self.path.removeprefix(proxy.prefix)
                if self.path != proxy.prefix + route or (self.command, route) not in {
                    ('GET', '/models'), ('POST', '/chat/completions'),
                }:
                    self.reply(404, b'{"error":{"message":"Unknown scoring API route"}}')
                    return
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 <= length <= 16 * 1024 * 1024:
                        raise ValueError('Invalid body length')
                except ValueError:
                    self.reply(400, b'{"error":{"message":"Invalid request body length"}}')
                    return
                content = self.rfile.read(length) if self.command == 'POST' else None
                headers = {key: self.headers[key] for key in ('Authorization', 'Content-Type', 'User-Agent')
                           if key in self.headers}
                headers['Accept-Encoding'] = 'identity'
                request = Request(proxy.upstream + route, data=content, headers=headers, method=self.command)
                try:
                    opener = build_opener(ProxyHandler({})) if urlsplit(proxy.upstream).hostname in {'127.0.0.1', 'localhost', '::1'} else build_opener()
                    with opener.open(request, timeout=proxy.timeout) as response:
                        status, response_headers, result = response.status, response.headers, response.read()
                except HTTPError as exc:
                    with exc:
                        status, response_headers, result = exc.code, exc.headers, exc.read()
                except (URLError, OSError):
                    self.reply(502, b'{"error":{"message":"Scoring upstream request failed"}}')
                    return
                if status == 200 and route == '/chat/completions':
                    try:
                        payload = json.loads(result)
                        if is_scoring_request(json.loads(content)) and isinstance(payload, dict):
                            changed = False
                            for choice in payload.get('choices', []):
                                message = choice.get('message') if isinstance(choice, dict) else None
                                if not isinstance(message, dict):
                                    continue
                                original = message.get('content')
                                normalized = normalize_score_response(original)
                                if normalized != original:
                                    message['content'] = normalized
                                    changed = True
                            if changed:
                                result = json.dumps(payload, ensure_ascii=False).encode('utf-8')
                                with proxy.lock:
                                    proxy.normalized_count += 1
                    except (ValueError, TypeError):
                        pass  # Preserve malformed responses for the original client's error handling.
                self.reply(status, result, response_headers.get('Content-Type', 'application/json'),
                           response_headers.get('Retry-After'))

            do_GET = forward
            do_POST = forward

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        self.base_url = f'http://127.0.0.1:{self.server.server_port}{self.prefix}'
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)
        print(f'[medflow] Score response compatibility: normalized {self.normalized_count} echoed responses', flush=True)


def argument_value(command, name, default=None):
    matches = [(index, item.partition('=')[2]) for index, item in enumerate(command)
               if item.startswith(name + '=')]
    matches += [(index, command[index + 1]) for index, item in enumerate(command[:-1]) if item == name]
    if len(matches) > 1:
        raise ValueError(f'Duplicate scoring argument: {name}')
    return matches[0] if matches else (None, default)


@contextmanager
def compatible_command(command):
    command = list(command)
    index, upstream = argument_value(command, '--base_url', 'http://localhost:8001/v1')
    _, timeout = argument_value(command, '--request-timeout', '60')
    timeout = float(timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Scoring request timeout must be positive and finite')
    with ScoringProxy(upstream, timeout) as proxy:
        if index is None:
            command.extend(['--base_url', proxy.base_url])
        elif command[index] == '--base_url':
            command[index + 1] = proxy.base_url
        else:
            command[index] = '--base_url=' + proxy.base_url
        yield command
