from typing import List, Dict, Any  # Added typing for clarity


def analyze_voice_authenticity_timeline(
    logs: List[Dict[str, Any]],
) -> Dict[str, Any]:  # Adjusted typing
    """Analyze voice authenticity across the entire pipeline."""

    # Group logs by frame_id
    frame_groups: Dict[str, List[Dict[str, Any]]] = {}  # Added typing
    for log in logs:
        frame_id = log.get("extra", {}).get(
            "frame_id"
        )  # Corrected: frame_id is usually under 'extra'
        # Also ensure that the log entry is relevant for voice analysis
        if frame_id and (
            log.get("message") == "VOICE_VERIFICATION"
            or log.get("message") == "VAD_VOICE_VERIFICATION"
            or log.get("message") == "RESAMPLING_VOICE_COMPARISON"
            or log.get("message")
            == "STT_CHUNK_INPUT_VOICE_VERIFICATION"  # Corrected log message name
            or log.get("message")
            == "VAD_INFERENCE_COMPLETE"  # Contains voice_verification dict
        ):
            if frame_id not in frame_groups:
                frame_groups[frame_id] = []
            frame_groups[frame_id].append(log)

    authenticity_analysis: Dict[str, Any] = {  # Added typing
        "total_frames_analyzed_for_voice": len(frame_groups),
        "human_voice_frames_final_assessment": 0,  # Clarified meaning
        "pipeline_voice_preservation_scores": [],  # Clarified meaning and type
        "rejection_patterns": {},
    }

    for frame_id, frame_logs in frame_groups.items():
        # Sort by timestamp_ns which should be in the 'extra' field
        sorted_logs = sorted(
            frame_logs, key=lambda x: x.get("extra", {}).get("timestamp_ns", 0)
        )

        # Track voice confidence through pipeline
        confidences = []
        locations = []
        is_human_at_stages = []

        for log_entry in sorted_logs:
            extra_data = log_entry.get("extra", {})
            confidence = None
            is_human = None
            location = extra_data.get("location")

            if log_entry.get("message") == "RESAMPLING_VOICE_COMPARISON":
                # For comparison, we might want to log both input and output,
                # but for a simple timeline, let's take output confidence here.
                confidence = extra_data.get("output_voice_confidence")
                # is_human could be inferred if output_voice_analysis.is_human_voice was logged
            elif log_entry.get("message") == "VAD_INFERENCE_COMPLETE":
                voice_ver_data = extra_data.get("voice_verification", {})
                confidence = voice_ver_data.get("confidence_score")
                is_human = voice_ver_data.get("is_human_voice")
            else:  # VOICE_VERIFICATION, VAD_VOICE_VERIFICATION, STT_CHUNK_INPUT_VOICE_VERIFICATION
                confidence = extra_data.get(
                    "confidence"
                )  # Plan used "confidence", but VoiceCharacteristics uses "confidence_score"
                if confidence is None:  # Fallback to correct field name
                    confidence = extra_data.get("confidence_score")
                is_human = extra_data.get("is_human_voice")

            if confidence is not None:
                confidences.append(confidence)
                locations.append(location)
            if is_human is not None:
                is_human_at_stages.append(is_human)

        if len(confidences) > 1:
            initial_confidence = confidences[0]
            final_confidence = confidences[-1]
            preservation_score = (
                final_confidence / initial_confidence if initial_confidence > 0 else 0
            )
            authenticity_analysis["pipeline_voice_preservation_scores"].append(
                preservation_score
            )

        if (
            is_human_at_stages and is_human_at_stages[-1]
        ):  # Check the last recorded human assessment for this frame_id
            authenticity_analysis["human_voice_frames_final_assessment"] += 1

        # Analyze rejection reasons (from any relevant log in the group for this frame_id)
        for log_entry in sorted_logs:
            extra_data = log_entry.get("extra", {})
            rejection_reasons = extra_data.get("rejection_reasons", [])
            for reason in rejection_reasons:
                authenticity_analysis["rejection_patterns"][reason] = (
                    authenticity_analysis["rejection_patterns"].get(reason, 0) + 1
                )

    return authenticity_analysis


def generate_performance_hotspots_report(
    logs: List[Dict[str, Any]],
) -> Dict[str, Any]:  # Adjusted typing
    """Identify performance bottlenecks from microscopic logs."""

    processing_times: Dict[str, List[float]] = {}  # location -> list of times_ms
    memory_spikes: Dict[str, List[float]] = {}  # location -> list of delta_mb
    network_issues: Dict[str, int] = {}  # location -> count

    for log_entry in logs:
        extra_data = log_entry.get("extra", {})
        location = extra_data.get("location", "unknown_location")
        level = log_entry.get("level", "INFO")  # Get log level

        # Processing time analysis from various duration fields
        # The plan used "processing_time_ns", but logs have specific duration fields e.g. "duration_ns", "operation_time_ns", "inference_latency_ms"
        duration_ns = None
        if "duration_ns" in extra_data:  # e.g. MEMORY_OPERATION, CHANNEL_OPERATION
            duration_ns = extra_data["duration_ns"]
        elif "operation_time_ns" in extra_data:  # e.g. RESAMPLING_OPERATION
            duration_ns = extra_data["operation_time_ns"]
        elif "total_duration_ns" in extra_data:  # e.g. PACKET_PROCESSING_COMPLETE
            duration_ns = extra_data["total_duration_ns"]
        elif "inference_latency_ms" in extra_data:  # e.g. VAD_INFERENCE_COMPLETE
            duration_ns = (
                extra_data["inference_latency_ms"] * 1_000_000
            )  # convert ms to ns
        elif "serialization_time_ns" in extra_data:  # e.g. STT_DATA_SERIALIZATION
            duration_ns = extra_data["serialization_time_ns"]
        elif "transmission_latency_ms" in extra_data:  # e.g. STT_NETWORK_TRANSMISSION
            duration_ns = (
                extra_data["transmission_latency_ms"] * 1_000_000
            )  # convert ms to ns

        if duration_ns is not None:
            processing_time_ms = duration_ns / 1_000_000
            if location not in processing_times:
                processing_times[location] = []
            processing_times[location].append(processing_time_ms)

        # Memory spike detection
        if "memory_delta_mb" in extra_data:
            memory_delta = extra_data["memory_delta_mb"]
            if memory_delta > 10:  # >10MB allocation or change considered a spike
                if location not in memory_spikes:
                    memory_spikes[location] = []
                memory_spikes[location].append(memory_delta)

        # Network performance issues (based on error logs for network-related locations)
        # This relies on location names containing "network" or specific error messages.
        if level == "ERROR" and (
            "network" in location.lower()
            or log_entry.get("message") == "STT_NETWORK_FAILURE"
            or log_entry.get("message") == "CHANNEL_SEND_ERROR"
            or log_entry.get("message") == "CHANNEL_QUEUE_FULL"
        ):
            network_issues[location] = network_issues.get(location, 0) + 1

    # Calculate statistics
    hotspots: Dict[str, Any] = {  # Added typing
        "slow_processing_operations": {},  # Renamed for clarity
        "memory_intensive_operations": {},  # Renamed for clarity
        "network_problem_summary": network_issues,  # Renamed for clarity
    }

    for loc, times in processing_times.items():
        if not times:
            continue
        avg_time = sum(times) / len(times)
        max_time = max(times)

        if (
            avg_time > 5 or max_time > 20
        ):  # Example thresholds: >5ms average or >20ms max considered slow
            hotspots["slow_processing_operations"][loc] = {
                "avg_time_ms": round(avg_time, 3),
                "max_time_ms": round(max_time, 3),
                "sample_count": len(times),
            }

    for loc, deltas in memory_spikes.items():
        if not deltas:
            continue
        avg_delta = sum(deltas) / len(deltas)
        max_delta = max(deltas)

        hotspots["memory_intensive_operations"][loc] = {
            "avg_allocation_mb": round(avg_delta, 3),
            "max_allocation_mb": round(max_delta, 3),
            "spike_count": len(deltas),
        }

    return hotspots
