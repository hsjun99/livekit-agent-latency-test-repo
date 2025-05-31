MICROSCOPIC_LOGGING_CONFIG = {
    "enabled": True,
    "voice_verification": {
        "enabled": True,
        "min_confidence_threshold": 0.6,
        "reject_non_human": False,  # Log but don't filter
        "detailed_analysis": True,
    },
    "memory_profiling": {
        "enabled": True,
        "snapshot_frequency": "every_frame",  # or "periodic"
        "track_allocations": True,
        "gc_monitoring": True,
    },
    "performance_profiling": {
        "enabled": True,
        "cpu_monitoring": True,
        "thread_tracking": True,
        "context_switch_monitoring": True,
    },
    "network_monitoring": {
        "enabled": True,
        "packet_level": True,
        "serialization_tracking": True,
        "transmission_timing": True,
    },
    "locations": {
        "webrtc_packet_received": {"enabled": True, "voice_verification": True},
        "resampling_operation": {"enabled": True, "voice_verification": True},
        "vad_inference_complete": {"enabled": True, "voice_verification": True},
        "stt_network_transmission": {"enabled": True, "voice_verification": True},
    },
}
