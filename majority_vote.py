from collections import deque, Counter
from typing import Optional, Hashable
import numpy as np


class MajorityVoteStabilizer:
    """
    Real-time majority-vote stabilizer for model predictions.

    This class receives gesture predictions one at a time and only commits
    a gesture when that gesture is the majority inside a recent rolling window.

    The middle layer is assumed to already filter out predictions below 70%
    confidence, so this class only sees high-confidence predictions.
    """

    def __init__(
        self,
        window_size: int = 7,
        majority_ratio: float = 0.60,
        min_votes: Optional[int] = None,
        require_full_window: bool = False,
        suppress_repeats: bool = True
    ):
        """
        Parameters
        ----------
        window_size:
            Number of recent high-confidence predictions to remember.

        majority_ratio:
            Fraction of the window that must agree before committing a gesture.
            For example, 0.60 with window_size=7 means at least 5 predictions
            must agree, because ceil(7 * 0.60) = 5.

        min_votes:
            Optional absolute vote requirement. If None, it is computed from
            window_size and majority_ratio.

        require_full_window:
            If True, the stabilizer waits until the rolling window is full
            before committing anything.
            If False, it can commit early once enough predictions agree.

        suppress_repeats:
            If True, the same committed gesture is not returned repeatedly.
            This is useful if you only want to trigger an action once per stable
            gesture instead of firing the same command every frame.
        """

        if window_size <= 0:
            raise ValueError("window_size must be positive.")

        if not 0.0 < majority_ratio <= 1.0:
            raise ValueError("majority_ratio must be between 0 and 1.")

        self.window_size = window_size
        self.majority_ratio = majority_ratio
        self.min_votes = min_votes
        self.require_full_window = require_full_window
        self.suppress_repeats = suppress_repeats

        self.prediction_buffer = deque(maxlen=window_size)
        self.last_committed_gesture = None

    def reset(self):
        """
        Clear the vote history.

        Call this when the user releases the gesture, when the system detects
        a long rest period, or when you want to start fresh.
        """
        self.prediction_buffer.clear()
        self.last_committed_gesture = None

    def update(self, predicted_gesture: Hashable) -> Optional[Hashable]:
        """
        Add one high-confidence prediction and return a committed gesture
        only if the majority agrees.

        Parameters
        ----------
        predicted_gesture:
            The gesture predicted by the model. This can be a string label
            such as "Fist", or an integer class ID such as 1.

        Returns
        -------
        committed_gesture:
            The stabilized gesture if the majority agrees.
            Otherwise returns None.
        """

        self.prediction_buffer.append(predicted_gesture)

        if self.require_full_window and len(self.prediction_buffer) < self.window_size:
            return None

        vote_counts = Counter(self.prediction_buffer)

        majority_gesture, majority_count = vote_counts.most_common(1)[0]

        if self.min_votes is None:
            required_votes = max(
                1,
                int(np.ceil(len(self.prediction_buffer) * self.majority_ratio))
            )
        else:
            required_votes = self.min_votes

        if majority_count < required_votes:
            return None

        if self.suppress_repeats and majority_gesture == self.last_committed_gesture:
            return None

        self.last_committed_gesture = majority_gesture
        return majority_gesture
    


vote_stabilizer = MajorityVoteStabilizer(
    window_size=7,
    majority_ratio=0.60,
    require_full_window=False,
    suppress_repeats=True
)




while (True): # Replace with actual prediction loop
    # predicted_gesture = model.predict(...)  # Get the latest high-confidence prediction
    predicted_gesture = "Fist"  # Placeholder for testing

    stabilized_gesture = vote_stabilizer.update(predicted_gesture)

    if stabilized_gesture is not None:
        print(f"Committed Gesture: {stabilized_gesture}")