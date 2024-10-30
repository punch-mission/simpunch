from simpunch.spike import generate_spike_image, load_spike_library


def test_random_spike_generation() -> None:
    """Randomly generate spike images to verify."""
    library = load_spike_library()
    for _ in range(100):
        spike_image = generate_spike_image((2048, 2048), library=library)
        assert spike_image.shape == (2048, 2048)
