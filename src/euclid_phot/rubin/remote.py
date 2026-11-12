"""Small RSP-side calculations through Rubin's documented notebook API.

Uses the same HTTP routes as rubin-nublado-client, without its Python>=3.13
requirement. Only PSF evaluation runs remotely; photometry runs locally.
https://nublado.lsst.io/user-guide/lab.html
"""
from __future__ import annotations

import json
import time
from urllib.parse import urljoin, urlsplit, urlunsplit
from uuid import uuid4
from html.parser import HTMLParser

import requests

from .client import validate_token
from ..netutils import ensure_https_ca_bundle


def spawn_options(html, size):
    """Use the current form's recommended image and actual available sizes."""
    class Inputs(HTMLParser):
        def __init__(self):
            super().__init__()
            self.inputs = []
        def handle_starttag(self, tag, attrs):
            if tag == 'input':
                self.inputs.append(dict(attrs))
    parser = Inputs()
    parser.feed(html)
    images = [v for v in parser.inputs if v.get('name') == 'image_list'
              and v.get('value', '').find('@sha256:') >= 0]
    selected = next((v for v in images if 'checked' in v), images[0] if images else None)
    sizes = {v.get('value') for v in parser.inputs if v.get('name') == 'size'}
    if not selected or size.lower() not in sizes:
        raise RuntimeError(f"Cannot select an RSP image/size from the spawn form; sizes: {sorted(sizes)}")
    return {'image_list': selected['value'], 'size': size.lower()}


class RSPExecutor:
    def __init__(self, token, *, site="https://data.lsst.cloud", size="Small"):
        ensure_https_ca_bundle()
        self._token = validate_token(token)
        self.site = site.rstrip("/")
        if urlsplit(self.site).scheme != 'https':
            raise ValueError('The RSP site must use HTTPS')
        self.size = size
        self.session = requests.Session()
        self.started_lab = False
        self.username = None
        self.hub = None
        self.lab = None
        self._hub_xsrf = self._lab_xsrf = None

    def _headers(self, lab=False):
        headers = {"Authorization": "Bearer " + self._token,
                   "Sec-Fetch-Mode": "same-origin", "Referer": self.hub + "/hub/home"}
        xsrf = self._lab_xsrf if lab else self._hub_xsrf
        if xsrf:
            headers["X-XSRFToken"] = xsrf
        return headers

    def _request(self, method, route, *, timeout=60, **kwargs):
        is_lab = route.startswith("user/")
        base = self.lab or self.hub if is_lab else self.hub
        url = base + "/" + route
        for _ in range(20):
            parsed = urlsplit(url)
            host = urlsplit(self.hub).hostname
            if parsed.scheme != "https" or parsed.hostname not in {host, self.username + "." + host}:
                raise RuntimeError("Notebook service redirected outside its trusted hostnames")
            headers = self._headers(is_lab)
            if method == "GET":
                headers["Sec-Fetch-Mode"] = "navigate"
            headers.update(kwargs.pop("headers", {}))
            response = self.session.request(method, url, headers=headers,
                                            allow_redirects=False, timeout=(30, timeout), **kwargs)
            xsrf = response.cookies.get("_xsrf")
            if xsrf:
                if "/user/" in parsed.path:
                    self._lab_xsrf = xsrf
                else:
                    self._hub_xsrf = xsrf
            if response.is_redirect:
                url = urljoin(url, response.headers["Location"])
                if is_lab and urlsplit(url).hostname == self.username + "." + host:
                    self.lab = urlunsplit(("https", urlsplit(url).netloc,
                                           urlsplit(self.hub).path, "", ""))
                if response.status_code in (302, 303):
                    method, kwargs = "GET", {}
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"RSP notebook API returned HTTP {response.status_code} for {route.split('/')[-1]}")
            return response
        raise RuntimeError("Too many notebook authentication redirects")

    def connect(self):
        headers = {"Authorization": "Bearer " + self._token}
        r = self.session.get(self.site + "/auth/api/v1/token-info", headers=headers, timeout=30)
        r.raise_for_status()
        info = r.json()
        if "exec:notebook" not in info.get("scopes", []):
            raise ValueError("Official PSF evaluation requires exec:notebook in addition to read:tap/read:image")
        self.username = info["username"]
        r = self.session.get(self.site + "/repertoire/discovery", timeout=30)
        r.raise_for_status()
        self.hub = r.json()["services"]["ui"]["nublado"]["url"].rstrip("/")
        expected_host = "nb." + urlsplit(self.site).hostname
        if urlsplit(self.hub).hostname != expected_host:
            raise RuntimeError("Unexpected notebook-service hostname in RSP discovery")
        self._request("GET", "hub/home")
        state = self._request("GET", "hub/api/users/" + self.username).json()
        if not state.get("servers"):
            form = self._request("GET", "hub/spawn")
            print("Starting a temporary RSP lab for official PSF evaluation", flush=True)
            self._request("POST", "hub/spawn", data=spawn_options(form.text, self.size))
            self.started_lab = True
        deadline = time.monotonic() + 240
        while True:
            state = self._request("GET", "hub/api/users/" + self.username).json()
            if any(server.get("ready") for server in state.get("servers", {}).values()):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("RSP lab did not become ready within four minutes")
            time.sleep(2)
        self._request("GET", "user/" + self.username + "/lab")
        self.lab = self.lab or self.hub
        return self

    def run_code(self, code, *, timeout=300):
        """Run a temporary LSST kernel via the supported execution extension."""
        notebook = {"cells": [{"cell_type": "code", "id": uuid4().hex,
                                "source": code, "metadata": {}, "outputs": [], "execution_count": None}],
                    "metadata": {"kernelspec": {"name": "lsst", "display_name": "LSST", "language": "python"}},
                    "nbformat": 4, "nbformat_minor": 5}
        response = self._request("POST", "user/" + self.username + "/rubin/execution",
                                  data=json.dumps(notebook), params={"kernel_name": "lsst"},
                                  headers={"X-Kernel-Name": "lsst"}, timeout=timeout)
        result = response.json()
        if result.get("error"):
            error = result["error"]
            message = error.get("evalue", error.get("err_msg", "Remote execution failed"))
            raise RuntimeError("RSP calculation failed: " + message.replace(self._token, "[REDACTED]"))
        executed = json.loads(result["notebook"])
        return "".join("".join(o.get("text", "")) for c in executed["cells"]
                       for o in c.get("outputs", []) if o.get("output_type") == "stream")

    def close(self):
        try:
            if self.started_lab:
                self._request("DELETE", "hub/api/users/" + self.username + "/server")
                self.started_lab = False
        finally:
            self.session.close()

    def __enter__(self):
        try:
            return self.connect()
        except Exception:
            self.close()
            raise

    def __exit__(self, *args):
        self.close()
