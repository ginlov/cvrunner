class MetricAggregator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
    
    def update(self, value: float):
        """Add a new value to the aggregator."""
        self.values.append(value)
    
    def count(self) -> int:
        return len(self.values)
    
    def sum(self) -> float:
        return sum(self.values)
    
    def average(self) -> float:
        return self.sum() / self.count() if self.count() > 0 else 0.0


class MultiMetricAggregator:
    def __init__(self):
        self.data = {}

    def reset(self):
        """Clear all stored metrics."""
        self.data = {}

    def update(self, metrics: dict):
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
        values = self.data.get(name, [])
        return sum(values) / len(values) if values else 0.0
    
    def summary(self) -> dict:
        return {k: sum(v)/len(v) for k, v in self.data.items() if v}
