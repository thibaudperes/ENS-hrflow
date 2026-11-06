"""Record the live demo page into reports/figures/g_demo.gif (uv run --group demo)."""

import io
import time
import urllib.request
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

URL = "https://hrflow-api-243544005364.europe-west1.run.app/"
OUT = Path("reports/figures/g_demo.gif")
VIEWPORT = {"width": 1100, "height": 820}
WIDTH = 760
MAX_BYTES = 1_000_000


def wake_up(timeout: float = 90) -> None:
    # The service scales to zero: the first request waits for a cold start of about 15 s.
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            with urllib.request.urlopen(URL + "demo/random", timeout=30) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(2)
    raise TimeoutError(URL)


def frame(page) -> Image.Image:
    image = Image.open(io.BytesIO(page.screenshot()))
    height = round(image.height * WIDTH / image.width)
    return image.resize((WIDTH, height), Image.LANCZOS).convert("RGB")


def show_session(page, frames: list, durations: list) -> None:
    page.wait_for_selector("#recs li")
    page.wait_for_timeout(300)
    frames.append(frame(page))
    durations.append(2500)
    bottom = page.evaluate("document.body.scrollHeight - window.innerHeight")
    step = max(bottom // 5, 1)
    for y in range(step, bottom + 1, step):
        page.evaluate(f"window.scrollTo(0, {y})")
        frames.append(frame(page))
        durations.append(180)
    durations[-1] = 2200
    page.evaluate("window.scrollTo(0, 0)")


def main() -> None:
    wake_up()
    frames, durations = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, color_scheme="light")
        page.goto(URL)
        show_session(page, frames, durations)
        page.click("#go")
        page.wait_for_function("!document.getElementById('go').disabled")
        show_session(page, frames, durations)
        browser.close()
    palette = [image.quantize(colors=40, method=Image.Quantize.MEDIANCUT) for image in frames]
    palette[0].save(
        OUT, save_all=True, append_images=palette[1:], duration=durations, loop=0, optimize=True
    )
    assert OUT.stat().st_size < MAX_BYTES, OUT.stat().st_size


if __name__ == "__main__":
    main()
