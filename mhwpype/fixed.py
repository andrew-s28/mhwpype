from datetime import datetime

from mhwpype.core import ReferencePeriod


class HOBDAY16:
    """Default recommendations for MHW analysis from Hobday et al. (2016)"""

    HALF_WINDOW_WIDTH = 5
    MINIMUM_EVENT_LENGTH = 5
    MAXIMUM_GAP_LENGTH = 2
    THRESHOLD = [0.90]  # noqa: RUF012


class FixedBaseline(ReferencePeriod):
    """
    Fixed baseline MHW analysis requires a reference period. The recommendations posited by Hobday et al. (2016) are
    the most commonly used settings for MHW identification and classification.
    """

    def __init__(self, reference_begin_datetime: datetime, reference_end_datetime: datetime) -> None:
        """
        The ReferencePeriod class is conveniently subclassed by the FixedBaseline class. The following functions
        are inherited and use the same inputs as the ReferencePeriod class.

        build_daily_climatology
        build_daily_threshold
        """
        self.methodology = "FixedBaseline"
        super().__init__(reference_begin_datetime, reference_end_datetime)
