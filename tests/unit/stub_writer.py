"""A review writer that needs no service, for tests: its text is a function of the brief."""

from review_writer import ReviewBrief


class StubReviewWriter:
    model = "stub"

    async def write(self, brief: ReviewBrief) -> str:
        return (f"The {brief.product_name} gets {brief.rating} stars from me. "
                f"Mostly about {brief.aspect.replace('_', ' ')}, in {brief.language}.")

    async def aclose(self) -> None:
        return None
