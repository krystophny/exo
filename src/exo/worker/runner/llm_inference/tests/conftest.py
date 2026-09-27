"""Dedicated memory guard tests never use Metal/GPU streams."""

import mlx.core as mx

mx.set_default_device(mx.cpu)
