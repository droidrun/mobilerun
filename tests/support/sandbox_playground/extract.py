"""Extract headlines and interview transcripts from a fictional news app."""

from datetime import datetime

from mobilerun_functions import Context
from pydantic import BaseModel, Field

from mobilerun.sandbox import cloud


class Input(BaseModel):
    max_pages: int = Field(default=3, ge=1, le=5)
    image_budget: int = Field(default=100_000, ge=0, le=200_000)


class Story(BaseModel):
    title: str
    source: str
    published_at: datetime
    transcript: str | None = None
    image: str | None = None


class Output(BaseModel):
    stories: list[Story]


@cloud("news-headlines")
def main(input: Input, ctx: Context) -> Output:
    import base64
    import io
    from datetime import datetime

    from PIL import Image

    device = ctx.device()
    stories = []
    issues = []
    seen = set()
    remaining = input.image_budget

    def visible():
        return [
            n for n in device.ui().nodes if not (n.get("hidden") or n.get("offscreen"))
        ]

    def feed_rows():
        nodes = visible()
        if not any(n.get("resource_id") == "feed" for n in nodes):
            raise RuntimeError("headlines screen unavailable")
        return {
            n["resource_id"]: n
            for n in nodes
            if (n.get("resource_id") or "").startswith("story:")
        }

    def capture():
        with Image.open(device.screenshot()) as source:
            image = source.convert("RGB")
            image.thumbnail((640, 640))
            for quality in (75, 55, 35):
                with io.BytesIO() as buffer:
                    image.save(buffer, format="JPEG", quality=quality, optimize=True)
                    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                if len(encoded) <= remaining:
                    return encoded
        return None

    def read_story(row):
        try:
            device.tap_node(row, humanize=False)
        except RuntimeError:
            issues.append("could not open story")
            return None

        try:
            fields = {n.get("resource_id"): n for n in visible()}
            if fields.get("title", {}).get("text") != row["text"]:
                issues.append("detail did not match the selected headline")
                return None
            story = Story(
                title=fields["title"]["text"],
                source=fields["source"]["text"],
                published_at=datetime.fromisoformat(fields["published"]["text"]),
            )
            try:
                story.image = capture()
                if story.image is None:
                    issues.append("image budget exceeded")
            except (OSError, RuntimeError):
                issues.append("screenshot unavailable")

            transcript_button = fields.get("show_transcript")
            if transcript_button:
                try:
                    device.tap_node(transcript_button, humanize=False)
                    try:
                        nodes = visible()
                        if not any(n.get("resource_id") == "transcript" for n in nodes):
                            raise RuntimeError("incorrect transcript screen")
                        story.transcript = "\n".join(
                            n["text"] for n in nodes if n.get("resource_id") == "line"
                        )
                    finally:
                        device.key("back")
                except (KeyError, RuntimeError):
                    issues.append("transcript unavailable")
            return story
        except (KeyError, TypeError, ValueError, RuntimeError):
            issues.append("could not read story detail")
            return None
        finally:
            device.key("back")

    try:
        device.start_app("com.example.news")
        for page in range(input.max_pages):
            keys = [key for key in feed_rows() if key not in seen]
            if not keys:
                break
            for key in keys:
                seen.add(key)
                row = feed_rows().get(key)
                if row is None:
                    issues.append("headline no longer visible")
                    continue
                story = read_story(row)
                if story is not None:
                    stories.append(story)
                    remaining -= len(story.image or "")
            if page + 1 < input.max_pages:
                device.swipe(180, 600, 180, 200, humanize=False)
    except RuntimeError:
        issues.append("headlines navigation stopped early")

    if issues:
        ctx.partial("; ".join(dict.fromkeys(issues)))
    return Output(stories=stories)
