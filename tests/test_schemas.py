from schemas import Chunk


def test_chunk_roundtrip():
    c = Chunk(chunk_id="a#0", source="a.md", title="A", section="Intro", text="hi")
    assert Chunk.model_validate(c.model_dump()) == c
