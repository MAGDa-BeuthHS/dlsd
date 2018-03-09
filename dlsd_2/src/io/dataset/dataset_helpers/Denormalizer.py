class Denormalizer:

    @staticmethod
    def normalize(data, max_value):
        return ((data / max_value) * .99999999) + 0.00000001

    @staticmethod
    def denormalize(data, max_value):
        return (data / .99999999) * max_value
