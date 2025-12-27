import inspect

class MetricAggregator:
    """
    A simple aggregator for collecting and summarizing metrics.
    """
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Clear all stored values."""
        self.values = []

    def update(self, value: float) -> None:
        """
        Add a new value to the aggregator.
        """
        self.values.append(value)

    def count(self) -> int:
        """
        Return the number of values stored.
        Returns:
            int: The count of values.
        """
        return len(self.values)

    def sum(self) -> float:
        """
        Return the sum of all stored values.
        Returns:
            float: The sum of values.
        """
        return sum(self.values)

    def average(self) -> float:
        """
        Return the average of all stored values.

        Returns:
            float: The average value, or 0.0 if no values are stored.
        """
        return self.sum() / self.count() if self.count() > 0 else 0.0


class MultiMetricAggregator:
    """
    An aggregator for multiple named metrics.
    """
    def __init__(self):
        self.data = {}

    def reset(self) -> None:
        """Clear all stored metrics."""
        self.data = {}

    def update(self, metrics: dict) -> None:
        """
        Update multiple metrics at once.

        Args:
            metrics (dict): {metric_name: value}
        """
        for name, value in metrics.items():
            if name not in self.data:
                self.data[name] = []
            self.data[name].append(value)

    def average(self, name: str) -> float:
        """
        Get the average of a specific metric.

        Args:
            name (str): The name of the metric.

        Returns:
            float: The average value of the metric, or 0.0 if not found.
        """
        values = self.data.get(name, [])
        return sum(values) / len(values) if values else 0.0

    def summary(self) -> dict:
        """
        Get the average of all stored metrics.

        Returns:
            dict: {metric_name: average_value}
        """
        return {k: sum(v)/len(v) for k, v in self.data.items() if v}

def get_properties(obj):
    """Extract all property names and values from an object"""
    properties = {}
    for name, value in inspect.getmembers(type(obj)):
        if isinstance(value, property):
            properties[name] = getattr(obj, name)
    return properties